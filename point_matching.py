import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import cv2

file = 'two_point.csv'
data = pd.read_csv(file)

world_points = np.array(data[['x', 'y', 'z']])
image_points = np.array(data[['x_1', 'y_1']])

# Add 1 to the 3D points
world_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

print(world_points)