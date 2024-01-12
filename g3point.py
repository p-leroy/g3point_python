import configparser
import os

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree

from g3point_python import tools
from g3point_python.detrend import rotate_point_cloud_plane, orient_normals
from g3point_python.segment import segment_labels
from g3point_python.visualization import show_clouds

#%% Inputs
dir_ = r"C:\DATA\PhilippeSteer\G3Point"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
cloud_detrended = os.path.join(dir_, "Otira_1cm_grains_rotated_detrended.ply")
ini = r"C:\dev\python\g3point_python\params.ini"

#%% Loading data
pcd_orig = o3d.io.read_point_cloud(cloud).points
xyz = np.asarray(pcd_orig)

#%% Algorithm parameters - Compute point cloud size and scaling of the algorithm
params = tools.read_parameters(ini)

#%% Denoise and decimate point cloud
pass

#%% Remove the points that are localized in local minima (multiscale) to ease segmentation and delimitation of grains
pass

#%% Rotate and detrend the point cloud (what about the polynomial trend?)
axis = np.array([0, 0, 1])
xyz_detrended = rotate_point_cloud_plane(xyz, axis)  # make the normal of the fitted plane aligned with the axis
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_detrended)

#%% Show the clean point cloud
# o3d.visualization.draw_geometries([pcd])

# Segment and cluster the point cloud into a point cloud of potential grains

#%% Find neighbors of point cloud
tree = KDTree(xyz_detrended)  # build a KD tree
neighbors_distances, neighbors_indexes = tree.query(xyz_detrended, params.knn + 1)
neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]

#%% Determine node surface
surface = np.pi * np.amin(neighbors_distances, axis=1) ** 2

#%% Compute normals and force them to point towards positive Z
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(params.knn))
centroid = np.mean(xyz_detrended, axis=0)
sensor_center = np.array([centroid[0], centroid[1], 1000])
normals = orient_normals(xyz_detrended, np.asarray(pcd.normals), sensor_center)

#%% Initial segmentation
labels, nlabels, labelsnpoint, stacks, ndon, sink_indexes = segment_labels(xyz_detrended, params.knn, neighbors_indexes)

#%% show initial segmentation
colors = np.random.rand(len(stacks), 3)[labels, :]
pcd.colors = o3d.utility.Vector3dVector(colors)
# build pcd_sinks
pcd_sinks = o3d.geometry.PointCloud()
pcd_sinks.points = o3d.utility.Vector3dVector(xyz_detrended[sink_indexes, :])
pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))

#%% Show clouds
clouds = (
    ('pcd', pcd, None, 3),
    ('pcd_sinks', pcd_sinks, None, 5)
)
show_clouds(clouds)