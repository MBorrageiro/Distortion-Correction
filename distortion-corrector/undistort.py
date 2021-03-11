import cv2
import numpy as np
import sys

# You should replace these 3 lines with the output in calibration step
DIM=(3840, 2160)
K=np.array([[2647.186781807643, 0.0, 1878.5274352779973], [0.0, 2669.2275927477576, 1195.2691387923333], [0.0, 0.0, 1.0]])
D=np.array([[0.4049844368850051], [0.30451980081006014], [0.6897057803481883], [-0.39927211675933677]])
def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)