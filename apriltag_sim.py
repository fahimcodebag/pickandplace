"""Virtual AprilTag perception for robosuite — tag injection, detection, pose.

Puts a real AprilTag into the simulated scene and recovers the object's 6-DoF
pose the same way the ESP32 would: render -> detect -> solvePnP -> world frame.
Because MuJoCo also knows the true pose, every estimate can be scored against
ground truth, which turns perception error into a measurable quantity instead
of an assumed one.

Design notes
------------
* Detection uses OpenCV's `DICT_APRILTAG_16h5`, selected by TAG_DICT below.
  Measured on the wrist camera at 320x240 (30 resets each), which is the
  deployment configuration:
      36h11   17/30 (57%)   pos 21.0 mm   ang 2.5 deg
      16h5    24/30 (80%)   pos 20.4 mm   ang 3.0 deg
  4x4 data bits survive a marginal ~19.7 px tag better than 6x6, at the same
  position accuracy. It is also the family the ESP32 port ships
  (stnk20/apriltag esp-idf branch carries tag16h5 and tag25h9, NOT tag36h11,
  whose 587-code table is expensive in flash).
  The cost is a weaker Hamming separation and so a higher false-positive rate;
  here there is exactly one tag and the upright-normal prior (min_up) already
  rejects implausible quads, but this needs watching on real imagery where
  noise produces far more spurious candidates than a clean render.
* The tag geom is VISUAL-ONLY (`contype=0 conaffinity=0 mass=0`). Physics must
  be bit-identical to the untagged environment or the 92%/90% results (§5, §8.4)
  stop being comparable.
* The texture carries a white quiet zone; AprilTag will not decode without one.
  `marker_size_m` accounts for the padding, and is what solvePnP is given.
* Frame conventions: robosuite's `get_camera_extrinsic_matrix` ALREADY applies
  the MuJoCo->OpenCV axis correction internally (diag(1,-1,-1)), so it returns
  a camera-to-world pose in OpenCV convention — exactly what solvePnP produces.
  No further flip is applied here. Adding one double-flips Y/Z and yields
  metre-scale errors that still look like valid poses (measured: 2.5 m at
  agentview, 0.46 m at the wrist camera).
"""

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

TAG_DICT = cv2.aruco.DICT_APRILTAG_16h5   # see module docstring


# ---------------------------------------------------------------------------
# Tag texture
# ---------------------------------------------------------------------------

def generate_tag_png(path, tag_id=0, marker_px=400, quiet_px=56):
    """Write a 36h11 tag with a white quiet zone.

    Returns `marker_fraction` — the marker's share of the image edge, needed to
    convert the geom's physical size into the marker's physical size.
    """
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(TAG_DICT), tag_id, marker_px)
    total = marker_px + 2 * quiet_px
    canvas = np.full((total, total), 255, dtype=np.uint8)
    canvas[quiet_px:quiet_px + marker_px, quiet_px:quiet_px + marker_px] = marker
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, canvas)
    return marker_px / total


# ---------------------------------------------------------------------------
# XML injection
# ---------------------------------------------------------------------------

def inject_tag(xml, tag_png, body_name="Bread_main", half_size=0.022,
               z_offset=0.026, tag_name="apriltag"):
    """Attach a visual-only textured tag plate to `body_name`.

    Args:
        half_size: half-edge of the square plate, metres (plate edge = 2x).
        z_offset:  height above the body origin, metres. Must clear the mesh
                   or the tag renders inside the object.
    """
    root = ET.fromstring(xml)

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "name": f"{tag_name}_tex", "type": "2d",
        "file": os.path.abspath(tag_png)})
    ET.SubElement(asset, "material", {
        "name": f"{tag_name}_mat", "texture": f"{tag_name}_tex",
        "texuniform": "false", "specular": "0", "shininess": "0",
        "reflectance": "0", "emission": "0.35"})

    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"body {body_name!r} not found in model XML")
    ET.SubElement(body, "geom", {
        "name": f"{tag_name}_geom", "type": "box",
        "size": f"{half_size} {half_size} 0.0012",
        "pos": f"0 0 {z_offset}",
        "material": f"{tag_name}_mat",
        # Visual-only: must not perturb dynamics (see module docstring).
        "contype": "0", "conaffinity": "0", "mass": "0", "group": "1",
    })
    return ET.tostring(root, encoding="unicode")


def make_tagged_env(env_name="PickPlace", camera="agentview", width=320,
                    height=240, tag_png=None, half_size=0.022, z_offset=0.026,
                    body_name="Bread_main", **make_kwargs):
    """robosuite env with a rendered AprilTag on the target object.

    Injection uses robosuite's own `set_xml_processor` hook, which
    `_initialize_sim()` applies on EVERY hard reset — so the tag survives
    `reset()` instead of being wiped by the next `_load_model()` call.

    Returns (env, meta); meta carries the marker's physical size and the camera
    settings `TagDetector` needs.
    """
    import robosuite as suite

    if tag_png is None:
        tag_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "assets", "tag36h11_0.png")
    marker_fraction = generate_tag_png(tag_png)
    marker_size_m = 2.0 * half_size * marker_fraction

    kwargs = dict(robots="Panda",
                  controller_configs=suite.load_controller_config(
                      default_controller="OSC_POSE"),
                  has_renderer=False, has_offscreen_renderer=True,
                  use_camera_obs=True, camera_names=camera,
                  camera_heights=height, camera_widths=width,
                  horizon=500, reward_shaping=False, control_freq=20,
                  single_object_mode=2, object_type="bread")
    kwargs.update(make_kwargs)
    env = suite.make(env_name, **kwargs)

    env.set_xml_processor(lambda xml: inject_tag(
        xml, tag_png, body_name=body_name,
        half_size=half_size, z_offset=z_offset))
    env.reset()

    meta = dict(marker_size_m=marker_size_m, camera=camera,
                width=width, height=height, tag_png=tag_png)
    return env, meta


# ---------------------------------------------------------------------------
# Detection + pose
# ---------------------------------------------------------------------------

class TagDetector:
    """Detects the tag in a rendered frame and returns its world-frame pose."""

    def __init__(self, env, camera="agentview", width=320, height=240,
                 marker_size_m=0.0344, tag_id=0, backend="opencv"):
        from robosuite.utils import camera_utils
        self._camera_utils = camera_utils
        self.env = env
        self.camera = camera
        self.width, self.height = width, height
        self.marker_size = marker_size_m
        self.tag_id = tag_id

        self.K = camera_utils.get_camera_intrinsic_matrix(
            env.sim, camera, height, width)
        self.dist = np.zeros(5)          # MuJoCo renders a pinhole camera

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(TAG_DICT),
            cv2.aruco.DetectorParameters())

        # Optional: use the ESP32 port (esp32_apriltag/) instead of OpenCV, so
        # the desktop numbers describe the detector the firmware actually runs
        # -- same C code, same ROI crop, same decimation. The two are NOT
        # interchangeable: corners differ by ~1 px, which moves the pose, and
        # the residual corrector is calibrated to whichever produced it.
        self.at32 = None
        if backend == "esp32":
            from at32 import At32Pose
            self.at32 = At32Pose()
        elif backend != "opencv":
            raise ValueError(f"unknown detector backend {backend!r}")

        h = marker_size_m / 2.0
        # Corner order must match OpenCV's: TL, TR, BR, BL.
        self.obj_pts = np.array([[-h, h, 0], [h, h, 0],
                                 [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    min_up = 0.30     # tag normal must point broadly upward (object rests flat)
    last_err = float("inf")   # reprojection error of the pose last returned
    last_corners = None       # 4x2 image corners of the pose last returned
    last_area_px = 0.0        # tag area in pixels (proxy for range/foreshortening)
    # Extrinsic calibration, measured over 30 detections at agentview 1280x960
    # against MuJoCo truth. The residual is fixed in the WORLD frame (sd
    # [11.0, 3.4, 10.8] mm) rather than the tag frame (sd [12.2, 13.5, 10.8]),
    # which identifies it as a camera-extrinsics/scale artifact and not a tag
    # mounting offset -- so it is corrected in world coordinates. On hardware
    # this is the same one-off calibration any fixed camera needs.
    # NOTE the z term also shows the true tag height is ~13 mm above the object
    # centre, not the 26 mm the z_offset default assumes.
    calib_world_m = np.array([-0.0147, -0.0047, 0.0133])
    # Optional learned residual correction. The constant offset above is the
    # zero-feature special case of exactly this model; a ridge fit on 2181
    # detections over runtime-available features cuts median error a further
    # 11.56 -> 4.58 mm (leave-one-seed-out, R^2 0.80/0.64/0.81). Linear, so it
    # is a handful of multiply-adds on the target. Off unless load_residual_model
    # is called, because it is a PER-SETUP calibration -- it encodes this
    # camera in this scene and must be refitted against real ground truth
    # before it means anything on hardware.
    residual_model = None

    def load_residual_model(self, path="assets/tag_residual_model.json"):
        import json
        m = json.load(open(path))
        self.residual_model = (m["features"], np.array(m["mu"]),
                               np.array(m["sd"]), np.array(m["W"]))
        return self

    def frame(self, obs_dict):
        """Extract the camera image, undoing robosuite's vertical flip."""
        return obs_dict[f"{self.camera}_image"][::-1]

    def detect(self, obs_dict):
        """Return (position_world[3], quat_world_xyzw[4]) or None if not seen."""
        img = self.frame(obs_dict)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.at32 is not None:
            # Full C perception front end: corners AND pose, the same code the
            # firmware runs. Its pose stage is AprilTag's orthogonal iteration,
            # not OpenCV IPPE; on identical corners the two agree to a median
            # of 0.04 mm (max 1.12), but it is the C one that ships, so it is
            # the C one that is measured and that the corrector is fitted to.
            r = self.at32.detect_pose(gray, self.K, self.marker_size,
                                      want_id=self.tag_id)
            if r is None or not r[1]:
                return None
            img_pts, at32_sols = r[0].astype(np.float32), r[1]
            corners, idx = [img_pts.reshape(1, 4, 2)], 0
        else:
            at32_sols = None
            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is None or self.tag_id not in ids.flatten():
                return None
            idx = int(np.where(ids.flatten() == self.tag_id)[0][0])

        # IPPE_SQUARE on a planar marker is TWO-fold ambiguous: two poses
        # project to nearly the same image. Taking solvePnP's single answer
        # gave a ~90 deg orientation error on 3 of 15 detections (measured:
        # median 3.8 deg but p95 91.3 deg), and one such pose held for a whole
        # perception period is enough to ruin the grasp -- end-to-end success
        # was 28% against 100% on ground truth.
        #
        # solvePnPGeneric returns BOTH solutions plus reprojection errors.
        # Disambiguate with physics rather than reprojection alone: before the
        # grasp the object rests on the table, so the tag's normal points UP.
        # The flipped solution points it away from world +z and is rejected.
        img_pts = corners[idx].reshape(4, 2).astype(np.float32)
        self.last_corners = img_pts.copy()
        # shoelace area: shrinks with range and with viewing obliquity, so it
        # is the natural single-number proxy for "how much tag do I actually
        # have to fit corners to"
        x, y = img_pts[:, 0], img_pts[:, 1]
        self.last_area_px = float(0.5 * abs(np.dot(x, np.roll(y, 1)) -
                                            np.dot(y, np.roll(x, 1))))
        if at32_sols is not None:
            n = len(at32_sols)
            errs_list = [e for _, e in at32_sols]
        else:
            n, rvecs, tvecs, err = cv2.solvePnPGeneric(
                self.obj_pts, img_pts, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            errs_list = None
        if not n:
            return None

        T_world_cam = self._camera_utils.get_camera_extrinsic_matrix(
            self.env.sim, self.camera)

        # Two-stage: the upright prior REJECTS the flipped solution, then
        # reprojection error CHOOSES among what survives. Scoring the two
        # together (up - k*err) picks upright poses that fit badly -- measured:
        # it fixed orientation (p95 91.3 -> 4.3 deg) but pushed position the
        # wrong way (median 9.3 -> 23.0 mm).
        errs = (np.asarray(errs_list).flatten() if errs_list is not None
                else np.asarray(err).flatten())
        cand = []
        for i in range(n):
            if at32_sols is not None:
                T_cam_tag = at32_sols[i][0]
            else:
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = cv2.Rodrigues(rvecs[i])[0]
                T_cam_tag[:3, 3] = np.asarray(tvecs[i]).flatten()
            T_wt = T_world_cam @ T_cam_tag
            up = float(T_wt[:3, 2] @ np.array([0.0, 0.0, 1.0]))
            cand.append((up, float(errs[i]) if i < len(errs) else np.inf, T_wt))
        upright = [c for c in cand if c[0] >= self.min_up]
        if not upright:
            return None
        chosen = min(upright, key=lambda c: c[1])
        # Expose the reprojection error so a caller with SEVERAL cameras can
        # pick the better view. Occlusion by the gripper shows up here: a
        # partially covered tag still decodes but localises its corners worse.
        self.last_err = chosen[1]
        best = chosen[2]
        return (best[:3, 3] - self.calib_world_m).copy(), _mat_to_quat(best[:3, :3])


def _mat_to_quat(R):
    """Rotation matrix -> xyzw quaternion (robosuite's convention)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, \
                     (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, \
                     (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, \
                     0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, \
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)
