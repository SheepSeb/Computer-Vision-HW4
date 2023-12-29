import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

file = 'data/points.csv'

pd.set_option('display.max_columns', None)
data = pd.read_csv(file)
data.head()

# Plot the points in a 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x'], data['y'], data['z'], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# Create a mesh from the points
points = np.array(data[['x', 'y', 'z']])
cloud = pv.PolyData(points)

# 3D Delaunay triangulation
surf = cloud.delaunay_3d(alpha=30)
surf.plot(show_edges=True, line_width=5)
