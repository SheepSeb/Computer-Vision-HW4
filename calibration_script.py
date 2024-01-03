import numpy as np
from scipy.optimize import leastsq
import pandas as pd
import cv2
import matplotlib.pyplot as plt

file = 'data/P1_calib.csv'
data = pd.read_csv(file)
data.head()

world_points = np.array(data[['x', 'y', 'z']],dtype=np.float32)
image_points = np.array(data[['x_1', 'y_1']],dtype=np.float32)
image_points_2 = np.array(data[['x_2', 'y_2']],dtype=np.float32)

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

def plot_points(P, image_points, image_points_proj):
    plt.figure()
    plt.scatter(image_points[:, 0], image_points[:, 1], color='blue', label='True Points')
    plt.scatter(image_points_proj[:, 0], image_points_proj[:, 1], color='red', label='Reprojected Points')
    # Add the movement of the points with arrows
    for i in range(len(image_points)):
        plt.arrow(image_points[i, 0], image_points[i, 1], image_points_proj[i, 0] - image_points[i, 0], image_points_proj[i, 1] - image_points[i, 1], color='green', width=0.0005)
    plt.title(f'True Points vs Reprojected Points with RMS Error: {np.mean(calculate_error(P, world_points, image_points)):.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


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
            cv2.putText(tmp_img, '({:.2f}, {:.2f}, {:.2f})'.format(world_points[i][0], world_points[i][1], world_points[i][2]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
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
            cv2.putText(img, '({:.2f}, {:.2f}, {:.2f})'.format(world_points[i][0], world_points[i][1], world_points[i][2]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        # Resize the image to 1280x720
        img = cv2.resize(img, (1280, 720))
        # Show the image
        cv2.imshow('image', img)
        cv2.waitKey(0)

def main():
    P1 = dlt(world_points, image_points)
    P2 = dlt(world_points, image_points_2)

    # Initial guess for P (reshape it to a 1D array)
    P1_initial = P1.ravel()
    P2_initial = P2.ravel()

    # Use least squares to refine P
    P1_opt, _ = leastsq(error_function, P1_initial, args=(world_points, image_points))
    P2_opt, _ = leastsq(error_function, P2_initial, args=(world_points, image_points_2))

    # Reshape P_opt from a 1D array to a 3x4 matrix
    P1_opt = P1_opt.reshape(3, 4)
    P2_opt = P2_opt.reshape(3, 4)
    
    image_points_proj_hom = np.dot(P1_opt, np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T).T
    image_points_proj = image_points_proj_hom[:, :2] / image_points_proj_hom[:, 2:]
    image_points_2_proj_hom = np.dot(P2_opt, np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T).T
    image_points_2_proj = image_points_2_proj_hom[:, :2] / image_points_2_proj_hom[:, 2:]
    plot_points(P1_opt, image_points, image_points_proj)
    plot_points(P2_opt, image_points_2, image_points_2_proj)
    
    # Show the images with the true and reprojected points
    img1 = cv2.imread('calibration_pictures/front.jpg')
    img2 = cv2.imread('calibration_pictures/side.jpg')
    
    view_img(img1, image_points, image_points_proj)
    view_img(img2, image_points_2, image_points_2_proj)
    
    # Save the two projection matrices
    np.savetxt('data/P1_new.txt', P1_opt)
    np.savetxt('data/P2_new.txt', P2_opt)
    
if __name__ == '__main__':
    main()