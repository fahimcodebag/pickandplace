"""Virtual AprilTag perception for robosuite — tag injection, detection, pose.

Puts a real AprilTag into the simulated scene and recovers the object's 6-DoF
pose the same way the ESP32 would: render -> detect -> solvePnP -> world frame.
Because MuJoCo also knows the true pose, every estimate can be scored against
ground truth, which turns perception error into a measurable quantity instead
of an assumed one.

Design notes
------------
* Detection uses OpenCV's `DICT_APRILTAG_36h11` — the same family the ESP32
  would run — so no extra dependency, and generation/detection stay consistent.
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

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11


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
                 marker_size_m=0.0344, tag_id=0):
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

        h = marker_size_m / 2.0
        # Corner order must match OpenCV's: TL, TR, BR, BL.
        self.obj_pts = np.array([[-h, h, 0], [h, h, 0],
                                 [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def frame(self, obs_dict):
        """Extract the camera image, undoing robosuite's vertical flip."""
        return obs_dict[f"{self.camera}_image"][::-1]

    def detect(self, obs_dict):
        """Return (position_world[3], quat_world_xyzw[4]) or None if not seen."""
        img = self.frame(obs_dict)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or self.tag_id not in ids.flatten():
            return None
        idx = int(np.where(ids.flatten() == self.tag_id)[0][0])

        ok, rvec, tvec = cv2.solvePnP(
            self.obj_pts, corners[idx].reshape(4, 2).astype(np.float32),
            self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None

        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cam_tag[:3, 3] = tvec.flatten()

        # Already OpenCV-convention camera-to-world (see module docstring).
        T_world_cam = self._camera_utils.get_camera_extrinsic_matrix(
            self.env.sim, self.camera)
        T_world_tag = T_world_cam @ T_cam_tag

        return T_world_tag[:3, 3].copy(), _mat_to_quat(T_world_tag[:3, :3])


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
