import cv2
import numpy as np
import glob

# -------- PARAMETERS --------
CHECKERBOARD = (6, 5)  # inner corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# -------- PREPARE OBJECT POINTS --------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) # 9*6 rows, 3 cols
# all rows, first two cols -> fill in x and y, not z 
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # creates two 2d arrays, represents checkboard layout

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob("calibration_images/*.jpg") # load calibrated images 

img_shape = None # making sure same size 

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_shape is None:
        img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) # ret: true/false, corners: pixel location of the detected corners. find checkerboard corners for each image 

    # if corners are found, use that image. if not, ignore it. 
    if ret:
        objpoints.append(objp)
        # making corner position more precise 
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Calibration", img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# process of estimating a camera's internal (intrinsic) and external (extrinsic) parameters to accurately map 3D real-world points to 2D image pixels, correcting for lens distortions and understanding the camera's position, enabling precise measurements, 3D reconstruction, and accurate computer vision tasks like robotics and augmented reality

# -------- CALIBRATION -------- 
# return value, camera matrix(most important), distortion coefficients, rotation vector, translation vector
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera( # camera matrix: how 3D points turn into 2D pixels, dist_coeffs: how camera lens distorts the image (bends light), rvecs: how much the camera is rotated for each calibrated image, tvecs: where the camera is located for each image 
    objpoints, imgpoints, img_shape, None, None
)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# -------- SAVE PARAMETERS --------
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

# -------- REPROJECTION ERROR --------
# tells you how accurate your camera calibration is 
total_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print("Mean reprojection error:", mean_error)