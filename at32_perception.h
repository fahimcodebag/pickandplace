// On-device AprilTag perception for the ESP32.
//
// This is the whole front end: pixels in, corrected object pose out. It is a
// line-by-line mirror of TagDetector.detect() in apriltag_sim.py, and the
// detector and pose solver underneath are literally the same C sources
// (src/esp32_apriltag/). What the PC still supplies is the camera-to-world
// transform, which is forward kinematics, not perception -- on a real arm the
// controller knows where the wrist camera is.
//
// Order matters and is not arbitrary; see apriltag_sim.py for the measurements
// behind each step:
//   1. detect the tag in the ROI crop
//   2. solve BOTH poses of the planar two-fold ambiguity
//   3. reject the one whose tag normal does not point broadly up (the object
//      rests flat before the grasp). Selecting on reprojection error alone put
//      a ~90 deg orientation error on 3 of 15 detections, and one such pose is
//      enough to ruin the grasp: 28% end-to-end against 100% on truth.
//   4. among the survivors take the lowest object-space error
//   5. subtract the fixed world-frame extrinsic calibration
//   6. apply the learned residual corrector (corrector_model.h)
#pragma once

#include <math.h>
#include <string.h>
#ifdef HOST_BUILD          // host verification build (see verify_perception.c)
#include "apriltag.h"
#include "tag16h5.h"
#include "apriltag_pose.h"
#include "common/image_u8.h"
#else
// Arduino: esp32_apriltag is a LIBRARY, not a sketch subfolder. Install it in
// Arduino/libraries/ so the IDE puts <lib>/src on the include path -- the
// upstream sources include each other as "common/xxx.h", which only resolves
// from there. Dropping the tree into the sketch's own src/ does NOT work.
#include <esp_heap_caps.h>   // heap_caps_get_largest_free_block
#include <apriltag.h>
#include <tag16h5.h>
#include <apriltag_pose.h>
#include <common/image_u8.h>
#endif

#define AT_ROI_W   222
#define AT_ROI_H   193
#define AT_ROI_X0   98
#define AT_ROI_Y0    0
#define AT_TAG_ID    0

// Wrist camera, 320x240, from robosuite's intrinsics. Full-frame pixels; the
// principal point is shifted into crop coordinates at use.
// Read from robosuite: get_camera_intrinsic_matrix(sim, "robot0_eye_in_hand",
// 240, 320). Do NOT guess these -- they set the pose scale.
static const double AT_FX = 156.387045, AT_FY = 156.387045;
static const double AT_CX = 160.0,      AT_CY = 120.0;
static const double AT_TAGSIZE = 0.034375;

// Fixed world-frame extrinsic calibration (apriltag_sim.py: calib_world_m).
static const float AT_CALIB[3] = {-0.0147f, -0.0047f, 0.0133f};
static const double AT_MIN_UP = 0.30;   // tag normal must point broadly up

typedef struct {
  float pos[3];        // corrected object position, world frame
  float quat[4];       // x,y,z,w
  float feat[12];      // corrector features, in corrector_model.h's order
  int   ok;
} at_result_t;

static apriltag_family_t   *at_tf = NULL;
static apriltag_detector_t *at_td = NULL;

// Reported rather than crashed on, so an out-of-memory shows up as OOM.
static void at_oom(const char* where){
#ifdef ARDUINO
  Serial.printf("[perc] OUT OF MEMORY at %s (free %u, largest block %u)\n",
                where, (unsigned)ESP.getFreeHeap(),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
#else
  (void)where;
#endif
}

static void at_perception_init(void){
  if(at_td) return;
  at_tf = tag16h5_create();
  at_td = apriltag_detector_create();
  apriltag_detector_add_family(at_td, at_tf);
  at_td->quad_decimate = 2.0f;   // ROI + decimate 2 is the memory-driven
  at_td->quad_sigma    = 0.0f;   // setting; see esp32_apriltag/README.md
  at_td->nthreads      = 1;
  at_td->refine_edges  = 1;
}

// R <- R * diag(-1,1,-1). AprilTag pairs ITS corner order with the same
// object points OpenCV uses, so the pose arrives in a tag frame rotated
// 180 deg about y. Raw the two disagree by a median of 142.7 deg; corrected,
// by 0.35 deg.
static void at_fix_frame(matd_t *R){
  for(int r=0;r<3;r++){
    MATD_EL(R, r, 0) = -MATD_EL(R, r, 0);
    MATD_EL(R, r, 2) = -MATD_EL(R, r, 2);
  }
}

static void at_mat_to_quat(const float R[9], float q[4]){
  float t = R[0] + R[4] + R[8];
  if(t > 0){ float s = sqrtf(t + 1.0f) * 2.0f;
    q[3] = 0.25f*s; q[0]=(R[7]-R[5])/s; q[1]=(R[2]-R[6])/s; q[2]=(R[3]-R[1])/s; }
  else if(R[0] > R[4] && R[0] > R[8]){ float s = sqrtf(1.0f+R[0]-R[4]-R[8])*2.0f;
    q[3]=(R[7]-R[5])/s; q[0]=0.25f*s; q[1]=(R[1]+R[3])/s; q[2]=(R[2]+R[6])/s; }
  else if(R[4] > R[8]){ float s = sqrtf(1.0f+R[4]-R[0]-R[8])*2.0f;
    q[3]=(R[2]-R[6])/s; q[0]=(R[1]+R[3])/s; q[1]=0.25f*s; q[2]=(R[5]+R[7])/s; }
  else { float s = sqrtf(1.0f+R[8]-R[0]-R[4])*2.0f;
    q[3]=(R[3]-R[1])/s; q[0]=(R[2]+R[6])/s; q[1]=(R[5]+R[7])/s; q[2]=0.25f*s; }
}

// roi: AT_ROI_W*AT_ROI_H grayscale. Twc: 3x4 row-major camera-to-world.
// eef: gripper position in world (the board already has it in its state).
static void at_perceive(const uint8_t *roi, const float *Twc,
                        const float *eef, at_result_t *out){
  out->ok = 0;
  at_perception_init();

  // The whole open question about this port is whether the detector fits in
  // internal SRAM. If it does not, the failure would otherwise be a silent
  // reboot mid-detection, which reads as a comms fault rather than as OOM.
  // Wrap the caller's buffer instead of copying it. image_u8_create would
  // allocate a SECOND 42 KB frame and hold both live through the whole
  // detection, on a board that reports only ~270 KB free.
  //
  // Safe because the detector never writes to its input at these settings:
  // quad_decimate = 2 > 1, so apriltag.c takes the image_u8_decimate() branch
  // and works on a fresh image, and quad_sigma = 0 skips the in-place blur
  // and the in-place sharpen. Read-only, hence the cast.
  // Positional, not designated: this header is included from the .ino and so
  // compiles as C++, where designated initializers are a GNU extension.
  // Fields are { width, height, stride, buf }.
  image_u8_t imv = { AT_ROI_W, AT_ROI_H, AT_ROI_W, (uint8_t*) roi };
  image_u8_t *im = &imv;
  zarray_t *dets = apriltag_detector_detect(at_td, im);
  if(!dets){ at_oom("apriltag_detector_detect"); return; }

  apriltag_detection_t *d = NULL;
  for(int i=0;i<zarray_size(dets);i++){
    apriltag_detection_t *c; zarray_get(dets, i, &c);
    // tag16h5 has only 30 codes and a small Hamming distance, so it throws
    // occasional false ids. Take only the tag we placed.
    if(c->id == AT_TAG_ID){ d = c; break; }
  }
  if(!d){ apriltag_detections_destroy(dets); return; }

  apriltag_detection_info_t info;
  info.det = d; info.tagsize = AT_TAGSIZE;
  info.fx = AT_FX; info.fy = AT_FY;
  info.cx = AT_CX - AT_ROI_X0; info.cy = AT_CY - AT_ROI_Y0;
  apriltag_pose_t s1, s2; double e1 = 0, e2 = 0;
  estimate_tag_pose_orthogonal_iteration(&info, &e1, &s1, &e2, &s2, 50);
  if(s1.R) at_fix_frame(s1.R);
  if(s2.R) at_fix_frame(s2.R);

  // Compose world <- tag for each candidate and keep the best UPRIGHT one.
  float bestR[9], bestT[3]; double bestErr = 0; int have = 0;
  apriltag_pose_t *S[2] = {&s1, &s2};
  double E[2] = {e1, e2};
  for(int k=0;k<2;k++){
    if(!S[k]->R) continue;
    float Rw[9], tw[3];
    for(int r=0;r<3;r++){
      for(int c=0;c<3;c++){
        double a = 0;
        for(int j=0;j<3;j++) a += Twc[r*4+j] * MATD_EL(S[k]->R, j, c);
        Rw[r*3+c] = (float)a;
      }
      double b = Twc[r*4+3];
      for(int j=0;j<3;j++) b += Twc[r*4+j] * MATD_EL(S[k]->t, j, 0);
      tw[r] = (float)b;
    }
    if(Rw[2*3+2] < AT_MIN_UP) continue;          // tag normal dot world +z
    if(!have || E[k] < bestErr){
      have = 1; bestErr = E[k];
      memcpy(bestR, Rw, sizeof bestR); memcpy(bestT, tw, sizeof bestT);
    }
  }
  if(s1.R){ matd_destroy(s1.R); matd_destroy(s1.t); }
  if(s2.R){ matd_destroy(s2.R); matd_destroy(s2.t); }

  if(have){
    // corners in FULL-frame pixels, AprilTag order -> OpenCV [1,0,3,2]
    static const int P[4] = {1,0,3,2};
    float px[4], py[4];
    for(int c=0;c<4;c++){
      px[c] = (float)(d->p[P[c]][0] + AT_ROI_X0);
      py[c] = (float)(d->p[P[c]][1] + AT_ROI_Y0);
    }
    float area = 0, cx = 0, cy = 0;
    for(int c=0;c<4;c++){
      int n = (c+3) & 3;                          // shoelace, matching numpy roll
      area += px[c]*py[n] - py[c]*px[n];
      cx += px[c]; cy += py[c];
    }
    area = fabsf(0.5f*area); cx *= 0.25f; cy *= 0.25f;

    float dp[3] = {bestT[0]-AT_CALIB[0], bestT[1]-AT_CALIB[1], bestT[2]-AT_CALIB[2]};
    const float campos[3] = {Twc[3], Twc[7], Twc[11]};
    // Every feature below is built from dp, the CALIBRATED detection -- that
    // is what wrist_dataset.py records, because it reads them off the value
    // detect() returns. Using the raw bestT here would silently shift the
    // whole feature distribution away from what the corrector was fitted on.
    float ray[3] = {dp[0]-campos[0], dp[1]-campos[1], dp[2]-campos[2]};
    float rng = sqrtf(ray[0]*ray[0]+ray[1]*ray[1]+ray[2]*ray[2]);
    float inv = 1.0f/(rng > 1e-9f ? rng : 1e-9f);
    ray[0]*=inv; ray[1]*=inv; ray[2]*=inv;

    float v[3] = {eef[0]-campos[0], eef[1]-campos[1], eef[2]-campos[2]};
    float vd = v[0]*ray[0]+v[1]*ray[1]+v[2]*ray[2];
    float pp[3] = {v[0]-vd*ray[0], v[1]-vd*ray[1], v[2]-vd*ray[2]};
    float perp = sqrtf(pp[0]*pp[0]+pp[1]*pp[1]+pp[2]*pp[2]);

    float nrm[3] = {bestR[2], bestR[5], bestR[8]};   // third COLUMN of R
    float ndr = fabsf(nrm[0]*ray[0]+nrm[1]*ray[1]+nrm[2]*ray[2]);
    if(ndr > 1.0f) ndr = 1.0f;
    float obliq = acosf(ndr) * 57.29577951308232f;

    at_mat_to_quat(bestR, out->quat);
    { float n2 = sqrtf(out->quat[0]*out->quat[0] + out->quat[1]*out->quat[1]
                     + out->quat[2]*out->quat[2] + out->quat[3]*out->quat[3]);
      if(n2 > 1e-12f) for(int c=0;c<4;c++) out->quat[c] /= n2; }
    float qx=out->quat[0], qy=out->quat[1], qz=out->quat[2], qw=out->quat[3];
    float yaw = atan2f(2.0f*(qw*qz + qx*qy), 1.0f - 2.0f*(qy*qy + qz*qz));

    out->feat[0]=(float)bestErr; out->feat[1]=area;   out->feat[2]=obliq;
    out->feat[3]=rng;            out->feat[4]=perp;   out->feat[5]=eef[2]-dp[2];
    out->feat[6]=cx;             out->feat[7]=cy;     out->feat[8]=yaw;
    out->feat[9]=dp[0];          out->feat[10]=dp[1]; out->feat[11]=dp[2];
    out->pos[0]=dp[0]; out->pos[1]=dp[1]; out->pos[2]=dp[2];
    out->ok = 1;
  }
  apriltag_detections_destroy(dets);
  // no image_u8_destroy: imv wraps the caller's buffer and owns nothing
}
