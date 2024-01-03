import numpy as np
import pandas as pd
import cv2
import pyvista as pv
import os
import matplotlib.pyplot as plt

# Load the matrix from the file
P1 = np.loadtxt('data/P1_new.txt')
P2 = np.loadtxt('data/P2_new.txt')

kpt1 = np.loadtxt('test/kpt0.txt')
kpt2 = np.loadtxt('test/kpt1.txt')

# Triangulate the points
points_3d = cv2.triangulatePoints(P1, P2, kpt1.T, kpt2.T).T

# Convert from homogeneous coordinates to Euclidean coordinates
points_3d_euclidean = points_3d[:, :3] / points_3d[:, 3:]


import open3d as o3d
def upsample_coordinates(coordinates, factor):
    # Convert the coordinates to a numpy array
    coords_np = np.array(coordinates)

    # Perform upsampling by linear interpolation
    upsampled_coords = []
    for i in range(len(coords_np) - 1):
        for j in range(factor):
            alpha = j / float(factor)
            interpolated_point = (1 - alpha) * coords_np[i] + alpha * coords_np[i + 1]
            upsampled_coords.append(interpolated_point)

    return np.asarray(upsampled_coords)

def visualize_point_clouds(original_coords, upsampled_coords):
    # Create Open3D point clouds from the coordinates
    original_point_cloud = o3d.geometry.PointCloud()
    original_point_cloud.points = o3d.utility.Vector3dVector(original_coords)

    upsampled_point_cloud = o3d.geometry.PointCloud()
    upsampled_point_cloud.points = o3d.utility.Vector3dVector(upsampled_coords)

    # Visualize the original and upsampled point clouds
    o3d.visualization.draw_geometries([original_point_cloud, upsampled_point_cloud])


original_point_cloud = o3d.geometry.PointCloud()
original_point_cloud.points = o3d.utility.Vector3dVector(points_3d_euclidean)

upsampled_coordinates = upsample_coordinates(points_3d_euclidean, factor=5)
visualize_point_clouds(points_3d_euclidean, upsampled_coordinates)
