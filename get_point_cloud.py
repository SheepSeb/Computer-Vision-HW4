import open3d as o3d
import numpy as np
import cv2

# Read a glb file with texture
# bunny = o3d.io.read_triangle_mesh("data/bunny.glb")
box = o3d.io.read_triangle_mesh("data/box_v2.glb")
bunny = o3d.io.read_triangle_mesh("data/bunny.glb", True)

# Put the bunny on the box and rotate it
bunny.translate((0, 0.45, 0))
R = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi,0))
bunny.rotate(R, center=(0, 0, 0))

picka = o3d.io.read_triangle_mesh("data/picka.glb",True)
mickey = o3d.io.read_triangle_mesh("data/mikey.glb",True)
picka.translate((-1.95, 0.15, -0.45))
picka.rotate(R, center=(0, 0, 0))
mickey.translate((1.45, 0, -1.65))
mickey.rotate(R, center=(0, 0, 0))
# Visualize the scene
o3d.visualization.draw_geometries([bunny,box])
o3d.visualization.draw_geometries([bunny,box,picka,mickey])