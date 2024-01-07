import open3d as o3d
import numpy as np

# Read a glb file with texture
bunny = o3d.io.read_triangle_mesh("data/bunny.glb")
box = o3d.io.read_triangle_mesh("data/box.glb")

# Put the bunny on the box and rotate it
bunny.translate((0, 0.4, 0))
R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2,0))
bunny.rotate(R, center=(0, 0, 0))

# Visualize the mesh
o3d.visualization.draw_geometries([bunny, box])