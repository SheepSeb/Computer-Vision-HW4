import numpy as np
import pandas as pd

# Load the matrix from the file
P1 = np.loadtxt('data/P1.txt')
P2 = np.loadtxt('data/P2.txt')

# The points to project

points_2d = pd.read_csv('data/keypoints.csv')
p1_points = np.array(points_2d[['x_1', 'y_1']], dtype=np.float32)
p2_points = np.array(points_2d[['x_2', 'y_2']], dtype=np.float32)

# Trinaguale the points in order to give the 3D points
# The 3D points are the same for both cameras

points_3d = np.zeros((len(p1_points), 4), dtype=np.float32)

for i in range(len(p1_points)):
    p1 = p1_points[i]
    p2 = p2_points[i]
    A = np.array([[p1[0]*P1[2, 0] - P1[0, 0], p1[0]*P1[2, 1] - P1[0, 1], p1[0]*P1[2, 2] - P1[0, 2], p1[0]*P1[2, 3] - P1[0, 3]],
                  [p1[1]*P1[2, 0] - P1[1, 0], p1[1]*P1[2, 1] - P1[1, 1], p1[1]*P1[2, 2] - P1[1, 2], p1[1]*P1[2, 3] - P1[1, 3]],
                  [p2[0]*P2[2, 0] - P2[0, 0], p2[0]*P2[2, 1] - P2[0, 1], p2[0]*P2[2, 2] - P2[0, 2], p2[0]*P2[2, 3] - P2[0, 3]],
                  [p2[1]*P2[2, 0] - P2[1, 0], p2[1]*P2[2, 1] - P2[1, 1], p2[1]*P2[2, 2] - P2[1, 2], p2[1]*P2[2, 3] - P2[1, 3]]])
    _, _, V = np.linalg.svd(A)
    points_3d[i] = V[-1]
    
points_3d = points_3d[:, :3] / points_3d[:, 3:]

# Print them as x,y,z coordinates with 3 decimals
for i in range(len(points_3d)):
    print('({:.3f}, {:.3f}, {:.3f})'.format(points_3d[i][0], points_3d[i][1], points_3d[i][2]))

# Plot the points
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()