import cv2
import numpy as np

# Load the images
img1 = cv2.imread('calibration_pictures/image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('calibration_pictures/image2.jpg', cv2.IMREAD_GRAYSCALE)

# Match the features
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Use FLANN to match the descriptors
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
# Draw the first 10 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=10)



# Show the image
# Resize the image to fit the screen
img3 = cv2.resize(img3, (0, 0), fx=0.2, fy=0.2)
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()