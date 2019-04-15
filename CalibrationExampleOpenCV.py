import numpy as np
import cv2
import glob

"""
@Author:            Enda McManemy  
@Student Number:    S00191109
@From:              Apr 2019  
@Code Derived from: https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html and
                    https://github.com/smidm/video2calibration/blob/master/calibrate.py
                    
                    Consider
                    https://github.com/kushalvyas/CameraCalibration/blob/master/src/intrinsic.py
                    which is based on https://kushalvyas.github.io/calib.html
                    
@Description:       POC Code to experiment with Camera Calibration using openCV and the provided images
"""

FilePath = 'C:\\CalibrationTest\\Images\\'
pattern_size = (7, 6)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(FilePath + '*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size,None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Chessboard Detected ")
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# objPoints (3d) and imgpoints (2d) now loaded - pass to function for calibration - It produces
# camera matrix, distortion coefficients, rotation and translation vectors
print('\nPerforming calibration...')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("RMS:", ret)
print("camera matrix:\n", mtx)
print("distortion coefficients: ", dist.ravel())

img2 = cv2.imread(FilePath + 'left11.jpg')
#print(img2)
h,  w = img2.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(FilePath + 'CAlibratedResult.png',dst)