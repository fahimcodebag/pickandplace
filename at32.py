"""ctypes bridge to the ESP32 AprilTag port (esp32_apriltag/).

Lets the desktop experiments run the EXACT detector the firmware runs, over
the exact same crop and decimation. That matters because the residual
corrector is a per-setup calibration and the detector is part of the setup:
a corrector fitted on OpenCV's full-resolution corners is not valid for the
device's corners.
"""
import ctypes, os
import numpy as np

# Crop window and decimation are the memory-driven deployment settings.
# The crop bound comes from n=2186 wrist detections (Results/wrist16_ds):
# cx 126.2..305.2, cy 13.0..164.2, max tag side 25.5 px, padded by 12 px.
ROI = (98, 0, 222, 193)     # x0, y0, w, h  within a 320x240 frame
DECIMATE = 2.0

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "esp32_apriltag", "libat32.so")


class _Det(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("p", ctypes.c_float * 8),
                ("cxy", ctypes.c_float * 2), ("hamming", ctypes.c_int),
                ("dm", ctypes.c_float)]


class At32Detector:
    def __init__(self, decimate=DECIMATE, sigma=0.0, refine_edges=1, roi=ROI):
        self.lib = ctypes.CDLL(_LIB)
        self.lib.at32_init.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int]
        self.lib.at32_detect.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_Det), ctypes.c_int]
        self.lib.at32_detect.restype = ctypes.c_int
        self.lib.at32_init(decimate, sigma, refine_edges)
        self.roi = roi

    def detect(self, gray, want_id=0):
        """Return the 4x2 corner array for `want_id`, in FULL-frame pixels, or None."""
        g = np.ascontiguousarray(gray, dtype=np.uint8)
        h, w = g.shape
        buf = (_Det * 8)()
        n = self.lib.at32_detect(
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), w, h,
            self.roi[0], self.roi[1], self.roi[2], self.roi[3], buf, 8)
        if n <= 0:
            return None
        for i in range(min(n, 8)):
            # tag16h5 has only 30 codes and a small Hamming distance, so it
            # throws occasional false IDs. Gate on the one tag we placed.
            if buf[i].id == want_id:
                p = np.array(buf[i].p, dtype=np.float64).reshape(4, 2)
                # AprilTag's corner order differs from OpenCV aruco's
                # TL,TR,BR,BL by the involution [1,0,3,2]. Measured on 39
                # frames where both detect: the permutation is the same on
                # 39/39, and under it per-corner agreement is median 0.95 px
                # (max 1.31). Everything downstream -- solvePnPGeneric, the
                # upright prior, the calibration -- assumes OpenCV order, so
                # reorder here rather than duplicating that logic.
                return p[[1, 0, 3, 2]]
        return None


class _Pose(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("p", ctypes.c_float * 8),
                ("R1", ctypes.c_float * 9), ("t1", ctypes.c_float * 3),
                ("e1", ctypes.c_float),
                ("R2", ctypes.c_float * 9), ("t2", ctypes.c_float * 3),
                ("e2", ctypes.c_float)]


class At32Pose(At32Detector):
    """Corners AND pose from the C port -- the whole perception front end.

    The firmware runs the pose stage on device too, so the desktop must run
    the same one. AprilTag's orthogonal iteration is NOT the same algorithm as
    OpenCV's IPPE_SQUARE and does not give the same pose, so a corrector
    fitted against IPPE is not valid for the device.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lib.at32_detect_pose.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, ctypes.POINTER(_Pose)]
        self.lib.at32_detect_pose.restype = ctypes.c_int

    def detect_pose(self, gray, K, tagsize, want_id=0):
        """Return (corners 4x2, [(T_cam_tag 4x4, err), ...]) or None."""
        g = np.ascontiguousarray(gray, dtype=np.uint8)
        h, w = g.shape
        out = _Pose()
        n = self.lib.at32_detect_pose(
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), w, h,
            self.roi[0], self.roi[1], self.roi[2], self.roi[3],
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
            float(tagsize), int(want_id), ctypes.byref(out))
        if n <= 0:
            return None
        corners = np.array(out.p, dtype=np.float64).reshape(4, 2)
        sols = []
        for R, t, e in ((out.R1, out.t1, out.e1), (out.R2, out.t2, out.e2)):
            if e < 0:
                continue
            T = np.eye(4)
            T[:3, :3] = np.array(R, dtype=np.float64).reshape(3, 3)
            T[:3, 3] = np.array(t, dtype=np.float64)
            sols.append((T, float(e)))
        return corners, sols
