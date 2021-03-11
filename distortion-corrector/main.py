import cv2
import os
import sys
import numpy as np
from IPython.display import Image


sys.path.append('./local-libs')

from calib import calib, app, utils

VIDEO_PATH = 'video-input/DJI_0002.MOV'
EXTRACTED_FRAMES_OUTPUT_DIR = 'extracted-frames'
INTRINSIC_DATA_DIR = 'intrinsic-data'
UNDIST_FRAMES_OUTPUT_DIR = 'undistorted-imgs'
POINTS_FILEPATH = os.path.join(INTRINSIC_DATA_DIR, "points.json")
CALIB_FILEPATH = os.path.join(INTRINSIC_DATA_DIR, "camera_calib.json")

def init_project_dirs():
    dirs = [EXTRACTED_FRAMES_OUTPUT_DIR, INTRINSIC_DATA_DIR, UNDIST_FRAMES_OUTPUT_DIR]
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

def extract_frames_from_video_input():
    # Opens the Video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        # Save every 30 frames
        if(not(i%30)):
            cv2.imwrite(os.path.join(EXTRACTED_FRAMES_OUTPUT_DIR, f'img{int(i):05}.jpg'),frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()
    print("Extraction Complete")

def extract_corners_from_images():
    # Camera & Board parameters
    camera_resolution = (3840,2160) # 4K

    board_edge_len = 0.0175 # 17.5mm
    board_shape = (9,6) # Row by columns

    # 'window_size' sets the size of the calibration board corner detector window size
    app.extract_corners_from_images(
        img_dir=EXTRACTED_FRAMES_OUTPUT_DIR,
        out_fpath=POINTS_FILEPATH,
        board_shape=board_shape,
        board_edge_len=board_edge_len,
        window_size=5,
        remove_unused_images=True
    )

    #Simply for illustration puposes
    app.plot_corners(POINTS_FILEPATH)

def calibrate_camera_intrinsics():
    K, D, R, t, used_points, rms = app.calibrate_fisheye_intrinsics(
        points_fpath=POINTS_FILEPATH,
        out_fpath=CALIB_FILEPATH
    )
    print(f"\nRMS Error is {rms:.3f} pixels")


if __name__=='__main__':
    print("Starting program")
    # init_project_dirs()
    # extract_frames_from_video_input()
    # extract_corners_from_images()
    calibrate_camera_intrinsics()

    scene = app.plot_points_fisheye_undistort(
        points_fpath=POINTS_FILEPATH,
        camera_fpath=CALIB_FILEPATH
    )

    Image(filename='./test-frames/img00540.jpg')

    # Load intrinsics
    k,d,resolution = utils.load_camera(CALIB_FILEPATH)

    # Img paths
    image_fpath = './test-frames/img00540.jpg'
    result_fpath = UNDIST_FRAMES_OUTPUT_DIR

    dist_img = cv2.imread(image_fpath)
    cv2.imwrite(os.path.join(result_fpath, "distorted_sample_img.jpg"), dist_img)

    # Undistort image
    result = cv2.fisheye.undistortImage(
        distorted=dist_img,
        K=k,
        D=d,
        Knew=k)

    cv2.imwrite(os.path.join(result_fpath, "undistored_sample_img.jpg"), result)

    # Undistort image and remove border - not working!
    k_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, resolution, np.eye(3), balance=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k_new, resolution, cv2.CV_16SC2)
    undist_img = cv2.remap(dist_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite(os.path.join(result_fpath,"undistored_sample_img_no_border.jpg"), undist_img)

    print("Program completed!")