import cv2
import numpy as np

# Load the images
img1 = cv2.imread('calibration_pictures/image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('calibration_pictures/image2.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the StereoSGBM matcher
window_size = 5
min_disp = 32
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 16,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

# Compute the disparity map
disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0

# Display the disparity map
cv2.imshow('Disparity Map', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()