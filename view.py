import numpy as np
import pandas as pd
import cv2
import pyvista as pv
import os
import matplotlib.pyplot as plt

# Read the generated np file
points = np.loadtxt('data/point_cloud_side.txt')

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])

# Create a mesh from the Delaunay triangulation using PyVista
mesh = pv.PolyData(points)
mesh = mesh.delaunay_3d(alpha=0.025)
mesh = mesh.texture_map_to_plane(inplace=True)
# texture = pv.read_texture('calibration_pictures/front.jpg')

# Plot the mesh
mesh.plot(show_edges=True, line_width=0.1, color='w', background='black')