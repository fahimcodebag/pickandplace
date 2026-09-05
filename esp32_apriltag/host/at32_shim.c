/* Thin wrapper exposing the ESP32 detection pipeline to the host.
 *
 * The POINT of this shim is that the desktop experiments and the ESP32 run
 * the SAME detector over the SAME crop at the SAME decimation. The corrector
 * is a per-setup calibration and the detector is part of that setup, so the
 * corrector must be fitted against these corners, not OpenCV's.
 */
#include <stdlib.h>
#include <string.h>
#include "apriltag.h"
#include "tag16h5.h"
#include "common/image_u8.h"

typedef struct { int id; float p[8]; float cxy[2]; int hamming; float dm; } at32_det_t;

static apriltag_family_t   *g_tf = NULL;
static apriltag_detector_t *g_td = NULL;

void at32_init(float decimate, float sigma, int refine_edges){
    if(g_td) return;
    g_tf = tag16h5_create();
    g_td = apriltag_detector_create();
    apriltag_detector_add_family(g_td, g_tf);
    g_td->quad_decimate = decimate;
    g_td->quad_sigma    = sigma;
    g_td->nthreads      = 1;          // the ESP32 build is single threaded
    g_td->refine_edges  = refine_edges;
}

/* gray: full-frame WxH, row-major, top-left origin.
 * The crop is applied here so the host sees exactly what the device sees;
 * returned corners are mapped BACK to full-frame coordinates. */
int at32_detect(const unsigned char *gray, int W, int H,
                int x0, int y0, int cw, int ch,
                at32_det_t *out, int maxout){
    if(!g_td) return -1;
    if(x0<0||y0<0||x0+cw>W||y0+ch>H) return -2;
    image_u8_t *im = image_u8_create(cw, ch);
    for(int y=0;y<ch;y++) memcpy(im->buf + y*im->stride, gray + (size_t)(y0+y)*W + x0, cw);
    zarray_t *dets = apriltag_detector_detect(g_td, im);
    int n = zarray_size(dets); if(n > maxout) n = maxout;
    for(int i=0;i<n;i++){
        apriltag_detection_t *d; zarray_get(dets, i, &d);
        out[i].id = d->id; out[i].hamming = d->hamming;
        out[i].dm = (float)d->decision_margin;
        for(int c=0;c<4;c++){
            out[i].p[2*c+0] = (float)(d->p[c][0] + x0);
            out[i].p[2*c+1] = (float)(d->p[c][1] + y0);
        }
        out[i].cxy[0] = (float)(d->c[0] + x0);
        out[i].cxy[1] = (float)(d->c[1] + y0);
    }
    int total = zarray_size(dets);
    apriltag_detections_destroy(dets);
    image_u8_destroy(im);
    return total;
}

/* ---- full perception: corners AND pose, exactly as the firmware does it ----
 *
 * The device must run the pose stage too, so the desktop has to run the SAME
 * pose stage or the corrector it fits is calibrated to the wrong thing.
 * estimate_tag_pose_orthogonal_iteration returns BOTH solutions of the planar
 * two-fold ambiguity together with their object-space errors, which is the
 * same structure cv2.solvePnPGeneric provides; the caller applies the upright
 * prior and picks by error, as before.
 */
#include "apriltag_pose.h"

typedef struct {
    int   id;
    float p[8];        /* corners, full-frame pixels, OpenCV TL,TR,BR,BL order */
    float R1[9], t1[3], e1;
    float R2[9], t2[3], e2;   /* e2 < 0 => no second solution */
} at32_pose_t;

/* R <- R * diag(-1,1,-1): negate columns 0 and 2. */
static void at32_fix_frame(apriltag_pose_t *ps){
    if(!ps || !ps->R) return;
    for(int r=0;r<3;r++){
        MATD_EL(ps->R, r, 0) = -MATD_EL(ps->R, r, 0);
        MATD_EL(ps->R, r, 2) = -MATD_EL(ps->R, r, 2);
    }
}

int at32_detect_pose(const unsigned char *gray, int W, int H,
                     int x0, int y0, int cw, int ch,
                     double fx, double fy, double cx, double cy,
                     double tagsize, int want_id, at32_pose_t *out){
    if(!g_td) return -1;
    if(x0<0||y0<0||x0+cw>W||y0+ch>H) return -2;
    image_u8_t *im = image_u8_create(cw, ch);
    for(int y=0;y<ch;y++) memcpy(im->buf + y*im->stride, gray + (size_t)(y0+y)*W + x0, cw);
    zarray_t *dets = apriltag_detector_detect(g_td, im);
    int found = 0;
    for(int i=0;i<zarray_size(dets) && !found;i++){
        apriltag_detection_t *d; zarray_get(dets, i, &d);
        if(d->id != want_id) continue;      /* tag16h5 throws false ids */
        found = 1;
        out->id = d->id;
        /* AprilTag order -> OpenCV aruco TL,TR,BR,BL is the involution
         * [1,0,3,2]; measured identical on 39/39 frames. */
        static const int P[4] = {1,0,3,2};
        for(int c=0;c<4;c++){
            out->p[2*c+0] = (float)(d->p[P[c]][0] + x0);
            out->p[2*c+1] = (float)(d->p[P[c]][1] + y0);
        }
        /* The detector ran on the CROP, so the principal point must be shifted
         * into crop coordinates -- the pose is estimated in that frame. */
        apriltag_detection_info_t info;
        info.det = d; info.tagsize = tagsize;
        info.fx = fx; info.fy = fy;
        info.cx = cx - x0; info.cy = cy - y0;
        apriltag_pose_t s1, s2; double e1=0, e2=0;
        estimate_tag_pose_orthogonal_iteration(&info, &e1, &s1, &e2, &s2, 50);
        /* AprilTag pairs ITS corner order with the same object points OpenCV
         * uses, and the two orders differ by the mirror [1,0,3,2]. The pose
         * therefore comes back in a tag frame rotated 180 deg about y from
         * the OpenCV/aruco convention: R_opencv = R_apriltag * diag(-1,1,-1).
         * Measured over 179 solutions -- raw the two disagree by a median of
         * 142.7 deg, after this correction by 0.35 deg (p90 1.53).
         * Applied HERE, in C, so the firmware and the desktop emit the same
         * convention and the upright prior and world calibration downstream
         * need no special case. */
        at32_fix_frame(&s1); if(s2.R) at32_fix_frame(&s2);
        out->e1 = (float)e1;
        for(int k=0;k<9;k++) out->R1[k] = (float)s1.R->data[k];
        for(int k=0;k<3;k++) out->t1[k] = (float)s1.t->data[k];
        matd_destroy(s1.R); matd_destroy(s1.t);
        if(s2.R){
            out->e2 = (float)e2;
            for(int k=0;k<9;k++) out->R2[k] = (float)s2.R->data[k];
            for(int k=0;k<3;k++) out->t2[k] = (float)s2.t->data[k];
            matd_destroy(s2.R); matd_destroy(s2.t);
        } else out->e2 = -1.0f;
    }
    apriltag_detections_destroy(dets);
    image_u8_destroy(im);
    return found;
}
