import numpy as np
from scipy.optimize import least_squares
import pandas as pd
import cv2

file = 'data/two_point.csv'
data = pd.read_csv(file)

world_points = np.array(data[['x', 'y', 'z']],dtype=np.float32)
image_points = np.array(data[['x_1', 'y_1']],dtype=np.float32)
image_points_2 = np.array(data[['x_2', 'y_2']],dtype=np.float32)

# Add 1 to the world points
world_points = np.hstack((world_points, np.ones((len(world_points), 1))))
image_points = np.hstack((image_points, np.ones((len(image_points), 1))))

print(world_points)
print(image_points)