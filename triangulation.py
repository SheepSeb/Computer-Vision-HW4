import numpy as np
import pandas as pd
import cv2
import pyvista as pv
import os
import matplotlib.pyplot as plt

# Load the matrix from the file
P1 = np.loadtxt('data/P1_new.txt')
P2 = np.loadtxt('data/P2_new.txt')

# The points are in a text file where the odd rows are for the P1 matrix and the even rows are for the P2 matrix
# Load the points from the file

file = 'data/points_test.txt'

# Read the file
p1_points = []
p2_points = []

with open(file) as f:
    for i, line in enumerate(f):
        if i % 2 == 0:
            p1_points.append(line.strip().split(','))
        else:
            p2_points.append(line.strip().split(','))
            
# Convert the points to float
p1_points = np.array(p1_points, dtype=np.float32)
p2_points = np.array(p2_points, dtype=np.float32)

print(p1_points.shape)
# Triangulate the points
points_3d = cv2.triangulatePoints(P1, P2, p1_points.T, p2_points.T)

# Convert the points to homogeneous coordinates
points_3d = points_3d / points_3d[3]

# Convert the points to cartesian coordinates
points_3d = points_3d[:3].T

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
plt.show()