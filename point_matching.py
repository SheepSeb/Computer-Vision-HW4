import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import cv2

img_file_name = 'calibration_pictures/image1.jpg'
img_2_file_name = 'calibration_pictures/image2.jpg'

img1 = cv2.imread(img_file_name)
img2 = cv2.imread(img_2_file_name)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

# Save the matches points to a file
# points_2d = pd.DataFrame(columns=['x_1', 'y_1', 'x_2', 'y_2'])
# for i in range(len(matches)):
#     points_2d.loc[i] = [kp1[matches[i].queryIdx].pt[0], kp1[matches[i].queryIdx].pt[1],
#                         kp2[matches[i].trainIdx].pt[0], kp2[matches[i].trainIdx].pt[1]]
# points_2d.to_csv('data/points_2d_t.csv', index=False)

# Draw the matches with a bigger line
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
# img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow('Matches', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute the dense optical flow using the Farneback method
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Get the coordinates of the points in the first image
y, x = np.mgrid[0:gray1.shape[0], 0:gray1.shape[1]]

# Add the flow to get the coordinates of the points in the second image
x2 = x + flow[..., 0]
y2 = y + flow[..., 1]

# Save the points to a file
points_2d = pd.DataFrame({'x_1': x.ravel(), 'y_1': y.ravel(), 'x_2': x2.ravel(), 'y_2': y2.ravel()})
points_2d.to_csv('data/points_2d_t.csv', index=False)