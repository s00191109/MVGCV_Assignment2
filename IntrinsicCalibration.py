import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

"""
@Authors:           Enda McManemy, Daniel Irving, Maurice Barry, Colum Jones  
@Facility           IT Sligo Engineering Students
@From:              Apr 2019  
@Code Derived from: This code has been formed using code examples derived from 
                    https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html and
                    https://github.com/smidm/video2calibration/blob/master/calibrate.py
                    https://github.com/kushalvyas/CameraCalibration/blob/master/src/intrinsic.py

@Description:       POC Code to experiment with Camera Calibration using openCV. Generates the 
                    Intrinsic matrix and distortion coefficients.
                    Generates a histogram and/or barchart of the reprojection errors to validate parameters
"""
# Replace this value with name of Experiment
Experiment = 'Experiment2'
# Replace this path with the path to the folder on local box containing the images to evaluate
FilePath = 'C:\\CalibrationTest3\\'
# Booleans to control displaying the identified points in images and charts
DISPLAY_IMAGES = 'TRUE'
DISPLAY_BAR_CHART = 'TRUE'
DISPLAY_HISTOGRAM = 'TRUE'
# The types of images to process
IMAGE_EXTENSIONS = '*.jpg'
# Represents the pattern size, in this case we used a 7 * 9 checkboard pattern
pattern_size = (7, 9)
# 20 represents the 20x20 squares of our chosen pattern
pattern_sq_size = 20
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, pattern_sq_size, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
# Array for the list of images processed
imgList = []
# dictionary for list of re-projection errors per image processed
dictError = {}
# variable for mean re-projection error
mean_error = 0
# variable for standard deviation reprojection error
sd = 0


def plat_reprojectionErrors_on_hist():
    """
    histogram of the reprojection of errors with standard deviation
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Reprojection errors')
    ax.set_ylabel('Frequency')
    print('sd is ', sd)
    print('mean error ', mean_error)
    n, bins, patches = plt.hist(list(dictError.values()), bins='auto')
    # add a 'best fit standard deviation line'
    y = ((1 / (np.sqrt(2 * np.pi) * sd)) *
         np.exp(-0.5 * (1 / sd * (bins - mean_error)) ** 2)) / 100
    ax.plot(bins, y, '--')
    ax.set_title(r'Histogram of ' + Experiment + ' : $\mu={:.4f}'.format(mean_error) + ',$\sigma={:.4f}'.format(sd))
    # axes = plt.gca()
    # axes.set_ylim([1,2])
    fig.tight_layout()
    plt.show()


def plot_parameters_on_barchart():
    """
    Bar Chart of the data
    """
    plt.bar(dictError.keys(), dictError.values(), 1.0, color='b')
    plt.ylabel('Mean Error in Pixels')
    plt.xlabel('Images')
    plt.title('Re-projection Error For ' + Experiment)
    plt.axhline(y=mean_error, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.5, 0.5, 'Mean: {:.4f}'.format(mean_error), fontsize=30, va='center', ha='center', backgroundcolor='w')
    plt.show()


def init_calibration(gray):
    """
    Iterates the images in the working directory, locating the chessboard pattern
    and building the object points and image points array
    It produces camera matrix (mtx), distortion coefficients (dist)
    """
    print('\nPerforming calibration...')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # TODO Check ret
    print("camera matrix:\n", mtx)
    print("distortion coefficients: ", dist.ravel())

    calculate_reprojectionErrors(mtx, dist, rvecs, tvecs)


def calculate_reprojectionErrors(mtx, dist, rvecs, tvecs):
    """
    Iterates the the object points and generates the reprojection of errors
    """
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error
        dictError[imgList[i]] = error

    global mean_error
    mean_error = tot_error / len(objpoints)
    print("Mean Reprojection Error: ", mean_error)

    global sd
    sd = np.std(list(dictError.values()))
    print("Stand Deviation of Reprojection Error: ", sd)


def process_sampleImages(images, objp):
    """
    Iterates the images in the working directory, locating the chessboard pattern
    and building the object points and image points array
    """
    for fname in images:
        #   track the names of files been processed in a dict
        imgList.append(fname[len(FilePath):100])
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Chessboard Detected in " + fname)
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            if DISPLAY_IMAGES == 'TRUE':
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                cv2.imshow(fname + 'Image with Pattern Corners', img)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

    init_calibration(gray)


def undistortImage(undistortImg, newFileName, mtx, dist):
    """
    Apply the Intrinsic matrix and distortion coeffs to the
    provided image to 'undistort'
    """
    img2 = cv2.imread(undistortImg)
    h, w = img2.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    cv2.initUndistortRectifyMap(mtx, dist, )

    # crop the image and write the undistorted image back to disk in supplied folder path
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # Dump the generated Image back to original folder
    cv2.imwrite(FilePath + newFileName, dst)


def main():
    """
    Main function.
    """

    if Experiment == "":
        print("Please specify a Title for this experiment using the Constant Experiment.")
    elif FilePath == "":
        print("Please specify an Image Folder using the Constant FilePath.")
    elif pattern_size == "":
        print("Please specify the size of the chosen pattern (e.g. 4*6) using the Constant pattern_size.")
    elif pattern_sq_size == "":
        print("Please specify the size of the squares in the chosen pattern using the Constant pattern_sq_size.")
    else:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((7 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

        images = glob.glob(FilePath + IMAGE_EXTENSIONS)
        process_sampleImages(images, objp)

        if DISPLAY_HISTOGRAM == 'TRUE':
            plat_reprojectionErrors_on_hist()

        if DISPLAY_BAR_CHART == 'TRUE':
            plot_parameters_on_barchart()


if __name__ == "__main__":
    main()
