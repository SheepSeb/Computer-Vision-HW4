import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import cv2

file = 'two_point.csv'
data = pd.read_csv(file)

world_points = np.array(data[['x', 'y', 'z']],dtype=np.float32)
image_points = np.array(data[['x_1', 'y_1']],dtype=np.float32)
image_points_2 = np.array(data[['x_2', 'y_2']],dtype=np.float32)

import numpy as np

def dlt(world_points, image_points):
    assert len(world_points) == len(image_points)
    assert len(world_points) >= 6

    A = []

    for i in range(len(world_points)):
        X, Y, Z = world_points[i]
        x, y = image_points[i]
        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x*X, x*Y, x*Z, x])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y*X, y*Y, y*Z, y])

    A = np.array(A)

    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape(3, 4)

    return P

# Usage:
P = dlt(world_points, image_points_2)
print(P)

from scipy.optimize import leastsq

def error_function(P, world_points, image_points):
    # Reshape P from a 1D array to a 3x4 matrix
    P = P.reshape(3, 4)

    # Convert world points from Euclidean to homogeneous
    world_points_hom = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

    # Project world points to 2D
    image_points_proj_hom = np.dot(P, world_points_hom.T).T
    image_points_proj = image_points_proj_hom[:, :2] / image_points_proj_hom[:, 2:]

    # Calculate error
    error = image_points - image_points_proj

    return error.ravel()

# Initial guess for P (reshape it to a 1D array)
P_initial = P.ravel()

# Use least squares to refine P
P_opt, _ = leastsq(error_function, P_initial, args=(world_points, image_points_2),xtol=1e-40, ftol=1e-40, maxfev=10000)

# Reshape P_opt from a 1D array to a 3x4 matrix
P_opt = P_opt.reshape(3, 4)

print(P_opt)

def calculate_error(P, world_points, image_points):
    num_points = world_points.shape[0]

    # Convert world points from Euclidean to homogeneous
    world_points_hom = np.hstack((world_points, np.ones((num_points, 1))))

    # Project world points to 2D
    image_points_proj_hom = np.dot(P, world_points_hom.T).T
    image_points_proj = image_points_proj_hom[:, :2] / image_points_proj_hom[:, 2:]

    # Calculate error
    error = np.sqrt(np.sum((image_points - image_points_proj)**2, axis=1))

    return error

import matplotlib.pyplot as plt

def plot_points(image_points, image_points_proj):
    plt.figure()
    plt.scatter(image_points[:, 0], image_points[:, 1], color='blue', label='True Points')
    plt.scatter(image_points_proj[:, 0], image_points_proj[:, 1], color='red', label='Reprojected Points')
    plt.title('True Points vs Reprojected Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Usage:
image_points_proj_hom = np.dot(P_opt, np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T).T
image_points_proj = image_points_proj_hom[:, :2] / image_points_proj_hom[:, 2:]
plot_points(image_points_2, image_points_proj)

# Plot the points on the image
img = cv2.imread('calibration_pictures/image2.jpg')

for i in range(len(image_points_2)):
    tmp_img = img.copy()
    tmp_img = cv2.circle(tmp_img, (int(image_points_2[i][0]), int(image_points_2[i][1])), 25, (255, 0, 255), -1)
    tmp_img = cv2.circle(tmp_img, (int(image_points_proj[i][0]), int(image_points_proj[i][1])), 25, (0, 255, 0), -1)
    # Add caption with the correct and reprojected points coordinates with big font
    cv2.putText(tmp_img, 'True: ({}, {})'.format(int(image_points_2[i][0]), int(image_points_2[i][1])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(tmp_img, 'Reprojected: ({}, {})'.format(int(image_points_proj[i][0]), int(image_points_proj[i][1])), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Add the 3D point coordinates with small font
    cv2.putText(tmp_img, '({:.2f}, {:.2f}, {:.2f})'.format(world_points[i][0], world_points[i][1], world_points[i][2]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    # Resize the image to 1280x720
    tmp_img = cv2.resize(tmp_img, (1280, 720))
    # Show the image
    cv2.imshow('image', tmp_img)
    cv2.waitKey(0)
    # Go to the next image
    cv2.destroyAllWindows()
    

# Resize the image to 1280x720
# img = cv2.resize(img, (1280, 720))
# cv2.imshow('image', img)
# cv2.waitKey(0)