# Distortion-Correction
This repo contains two notebooks with stepped processes to undistort images and then further evaluate the effectveness of the undistortion process.

## Distortion Requirements PDF
This PDF document contains details that give an overview of the requirements of the undistortion process. Included is a suggested process for creating checkerboard calibration videos.

## Distortion Corrector
This notebook outlines the step by step process of determining the intrinsic parameters of a camera and using them to correct the distortion associated.

Work Flow:
1. Extract varying frames of a checkerboard from a calibration video,
2. Identify the pixel coordinates of corners of the checkerboard in each frame,
3. Use the pixel coordinates and knowns to determine the intrinsic parameters of the camera.
4. Given the intrinsic parameters the distortion coefficients can be used to undistort individual frames.

Files in this module:
- A notebooks detailing the steps to undistort images
- The 'Calib' folder which contains a number of helper functions that play a part in the unditortion process, they also deal with preprocssing of data before input to OpenCV functions.
- Image folders with examples as well as example points and camera parameter json files.
- A requirements.txt with the necessary packages for notebook.

## Distortion Comparison
A notebook that illustrates the effectiveness of the undistortion process. The process includes fitting ellipses to spheres viewed in the camera scene, 
comparing them before and after undistortion and finally comparing the ellipses to a unit circle.

Work Flow:
1. Segment the image based on colour to determine the regions of the image which contain the spheres.
2. Create estimated contours in the regions likely to contain a sphere.
3. Refine the contours and determine which ones actually represent spheres, these regions then have ellispes fitted to them.
4. Having completed the above steps for images before and after distortion, the resulting ellipses are compared to each other and a unit circle to illustrate the improvement and effectiveness of the undistortion.

Files in this module:
- A notebook with detailed steps on the distortion comparison,
- Two example images
