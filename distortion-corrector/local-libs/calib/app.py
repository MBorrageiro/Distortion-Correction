import os
import numpy as np
import json
from pprint import pprint

from .calib import calibrate_fisheye_camera, \
    create_undistort_fisheye_point_function

from .points import find_corners_images

from .utils import create_board_object_pts, \
    save_points, load_points, \
    save_camera, load_camera

from .plotting import plot_calib_board


def extract_corners_from_images(img_dir, out_fpath, board_shape, board_edge_len, window_size=11, remove_unused_images=False):
    print(f"Finding calibration board corners for images in {img_dir}")
    filepaths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(".jpg") or fname.endswith(".png")])
    points, fpaths, camera_resolution = find_corners_images(filepaths, board_shape, window_size=window_size)
    saved_fnames = [os.path.basename(f) for f in fpaths]
    saved_points = points.tolist()
    if remove_unused_images:
        for f in filepaths:
            if os.path.basename(f) not in saved_fnames:
                print(f"Removing {f}")
                os.remove(f)
    save_points(out_fpath, saved_points, saved_fnames, board_shape, board_edge_len, camera_resolution)


def plot_corners(points_fpath):
    points, fnames, board_shape, board_edge_len, camera_resolution = load_points(points_fpath)
    plot_calib_board(points, board_shape, camera_resolution)



def plot_points_fisheye_undistort(points_fpath, camera_fpath):
    k, d, camera_resolution = load_camera(camera_fpath)
    points, _, board_shape, *_ = load_points(points_fpath)
    undistort_pts = create_undistort_fisheye_point_function(k, d)
    undistorted_points = undistort_pts(points).reshape(points.shape)
    plot_calib_board(undistorted_points, board_shape, camera_resolution)


def calibrate_fisheye_intrinsics(points_fpath, out_fpath):
    points, fnames, board_shape, board_edge_len, camera_resolution = load_points(points_fpath)
    obj_pts = create_board_object_pts(board_shape, board_edge_len)
    k, d, r, t, used_points, rms = calibrate_fisheye_camera(obj_pts, points, camera_resolution)
    print("K:\n", k, "\nD:\n", d)
    save_camera(out_fpath, camera_resolution, k, d)
    return k, d, r, t, used_points, rms
