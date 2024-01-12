import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from g3point_python import tools
from g3point_python.detrend import orient_normals
from g3point_python.segment import segment_labels
from g3point_python.visualization import show_clouds

from lidar_platform import sbf

# Inputs
dir_ = r"C:\DATA\PhilippeSteer\G3Point"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
ini = r"C:\dev\python\g3point_python\params.ini"

# Load data
pcd_orig = o3d.io.read_point_cloud(cloud).points
xyz = np.asarray(pcd_orig)

params = tools.read_parameters(ini)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Find neighbors of point cloud
tree = KDTree(xyz)  # build a KD tree
neighbors_distances, neighbors_indexes = tree.query(xyz, params.knn + 1)
neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]

# Compute normals and force them to point towards positive Z
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(params.knn))
centroid = np.mean(xyz, axis=0)
sensor_center = np.array([centroid[0], centroid[1], 1000])
normals = orient_normals(xyz, np.asarray(pcd.normals), sensor_center)

# Initial segmentation
labels, nlabels, labelsnpoint, stacks, ndon, sink_indexes = segment_labels(xyz, params.knn, neighbors_indexes)

# set pcd random colors
rng = np.random.default_rng(42)
colors = rng.random((len(stacks), 3))[labels, :]
pcd.colors = o3d.utility.Vector3dVector(colors)

# build pcd_sinks
pcd_sinks = o3d.geometry.PointCloud()
pcd_sinks.points = o3d.utility.Vector3dVector(xyz[sink_indexes, :])
pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))

#%%
clouds = (
    ('pcd', pcd, None, 3),
    ('pcd_sinks', pcd_sinks, None, 5)
)
show_clouds(clouds)

filename = os.path.join(dir_, 'with_labels.sbf')
sbf.write_sbf(filename, xyz, labels)
