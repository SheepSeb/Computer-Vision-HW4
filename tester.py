import cv2
import numpy as np

# Load the images
img1 = cv2.imread('calibration_pictures/front.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('calibration_pictures/side.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Initialize the BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Get the matching points
points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Now points1 and points2 contain the matching points in the two images
# Show the matching points
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2, matchesThickness=10)
img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()