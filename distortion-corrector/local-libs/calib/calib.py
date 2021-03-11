import numpy as np
from typing import Tuple, Union
from nptyping import Array
import cv2
#from .utils import create_board_object_pts
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import time
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint


# ========== FISHEYE CAMERA MODEL ==========
def calibrate_fisheye_camera(obj_pts: Array[np.float32, ..., 3], img_pts: Array[np.float32, ..., ..., 2],
                             camera_resolution: Tuple[int, int]) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                    np.ndarray, Array[np.float32, ..., ..., 2]], None]:
    assert len(img_pts) >= 4, "Need at least 4 vaild frames to perform calibration."
    obj_pts_new = np.repeat(obj_pts[np.newaxis, :, :], img_pts.shape[0], axis=0).reshape((img_pts.shape[0], -1, 1, 3))
    img_pts_new = img_pts.reshape((img_pts.shape[0], -1, 1, 2))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)
    try:
        ret, k, d, r, t = cv2.fisheye.calibrate(obj_pts_new, img_pts_new, camera_resolution, None, None, None, None, flags,
                                                criteria)
        if ret:
            return k, d, r, t, img_pts, ret
    except Exception as e:
        if "CALIB_CHECK_COND" in str(e):
            idx = int(str(e)[str(e).find("input array ") + 12:].split(" ")[0])
            print(f"Image points at index {idx} caused an ill-conditioned matrix")
            img_pts = img_pts[np.arange(len(img_pts)) != idx]
            return calibrate_fisheye_camera(obj_pts, img_pts, camera_resolution)


def create_undistort_fisheye_point_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...]):
    def undistort_points(pts: Array[np.float32, ..., 2]):
        pts = pts.reshape((-1, 1, 2))
        undistorted = cv2.fisheye.undistortPoints(pts, k, d, P=k)
        return undistorted.reshape((-1,2))
    return undistort_points


def create_undistort_fisheye_img_function(k: Array[np.float64, 3, 3], d: Array[np.float64, ...], camera_resolution):
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, camera_resolution, cv2.CV_32FC1)
    def undistort_image(img):
        dst = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return dst
    return undistort_image
