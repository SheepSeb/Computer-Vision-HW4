import cv2
import pandas as pd
import numpy as np

keypoints = pd.read_csv('data/keypoints.csv')

img1 = cv2.imread('calibration_pictures/image2.jpg')
P2 = np.loadtxt('data/P2.txt')

def view_img(img, point, reprojects, one_by_one=False):
    n = len(point)
    if one_by_one == True:
        for i in range(n):
            tmp_img = img.copy()
            tmp_img = cv2.circle(tmp_img, (int(point[i][0]), int(point[i][1])), 25, (255, 0, 255), -1)
            tmp_img = cv2.circle(tmp_img, (int(reprojects[i][0]), int(reprojects[i][1])), 25, (0, 255, 0), -1)
            # Add caption with the correct and reprojected points coordinates with big font
            cv2.putText(tmp_img, 'True: ({}, {})'.format(int(point[i][0]), int(point[i][1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(tmp_img, 'Reprojected: ({}, {})'.format(int(reprojects[i][0]), int(reprojects[i][1])), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Add the 3D point coordinates with small font
            # Resize the image to 1280x720
            tmp_img = cv2.resize(tmp_img, (1280, 720))
            # Show the image
            cv2.imshow('image', tmp_img)
            cv2.waitKey(0)
            # Go to the next image
            cv2.destroyAllWindows()
    else:
        for i in range(n):
            cv2.circle(img, (int(point[i][0]), int(point[i][1])), 25, (255, 0, 255), -1)
            cv2.circle(img, (int(reprojects[i][0]), int(reprojects[i][1])), 25, (0, 255, 0), -1)
            # Add caption with the correct and reprojected points coordinates with big font
            cv2.putText(img, 'True: ({}, {})'.format(int(point[i][0]), int(point[i][1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(img, 'Reprojected: ({}, {})'.format(int(reprojects[i][0]), int(reprojects[i][1])), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Add the 3D point coordinates with small font
        # Resize the image to 1280x720
        img = cv2.resize(img, (1280, 720))
        # Show the image
        cv2.imshow('image', img)
        cv2.waitKey(0)

for i in range(len(keypoints)):
    x = int(keypoints['x_2'][i])
    y = int(keypoints['y_2'][i])
    cv2.circle(img1, (x, y), 25, (0, 0, 255), -1)



# Resize to 1280x720
img1 = cv2.resize(img1, (1280, 720))
cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

view_img(img1, np.array(keypoints[['x_2', 'y_2']], dtype=np.float32), np.array(keypoints[['x_2', 'y_2']], dtype=np.float32), one_by_one=True)