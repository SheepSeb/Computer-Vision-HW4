import numpy as np
import pandas as pd
import cv2
import pyvista as pv
import os
import matplotlib.pyplot as plt

# Load the matrix from the file
P1 = np.loadtxt('data/P1_new.txt')
P2 = np.loadtxt('data/P2_new.txt')

kpt1 = np.loadtxt('data/p1_points.txt')
kpt2 = np.loadtxt('data/p2_points.txt')

# Triangulate the points
points_3d = cv2.triangulatePoints(P1, P2, kpt1.T, kpt2.T).T

# Convert from homogeneous coordinates to Euclidean coordinates
points_3d_euclidean = points_3d[:, :3] / points_3d[:, 3:]

from scipy.spatial import ConvexHull,Delaunay

# Create a Delaunay triangulation
tri = ConvexHull(points_3d_euclidean)
triangulation = Delaunay(points_3d_euclidean)

# Plot the mesh
cloud = pv.PolyData(points_3d_euclidean)
cloud.plot(show_edges=True, line_width=0.1, color='w', background='black')

# Create a mesh from the Delaunay triangulation using PyVista
mesh = cloud.delaunay_3d(alpha=1)
mesh = mesh.texture_map_to_plane(inplace=True)
texture = pv.read_texture('calibration_pictures/front.jpg')
mesh.plot(show_edges=True,texture=texture, line_width=0.1, color='w', background='black')
