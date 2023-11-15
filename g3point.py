import configparser
import os

import numpy as np
import open3d as o3d

from g3point_python import tools
from g3point_python.fit_plane import adjust_normal_3d, fit_plane, vec2rot

#%% Inputs
dir_ = r"C:\DATA\PhilippeSteer\G3Point"
cloud = os.path.join(dir_, "Otira_1cm_grains_rotated.ply")
cloud_detrended = os.path.join(dir_, "Otira_1cm_grains_rotated_detrended.ply")
ini = r"C:\dev\python\g3point_python\params.ini"

#%% Loading data
pcd_orig = o3d.io.read_point_cloud(cloud).points
xyz = np.asarray(pcd_orig)

#%% Algorithm parameters - Compute point cloud size and scaling of the algorithm
params = tools.read_parameters(ini)

#%% Denoise and decimate point cloud

#%% Remove the points that are localized in local minima (multiscale) to ease segmentation and delimitation of grains

#%% Rotate and detrend the point cloud (what about the polynomial trend?)

a, b, c, dist_signed = fit_plane(xyz)  # fit a plane to the data and get the normal
normal_raw = np.array([-b, -a, 1])

point0 = np.array([0, 0, 0])
sensor_center = np.array([0, 0, 1e32])

normal = adjust_normal_3d(point0, normal_raw, sensor_center)

rotation = vec2rot(normal, np.array([0, 0, 1]))
centroid = np.mean(xyz, axis=0)
xyz_detrended = (rotation @ (xyz - centroid).T).T + centroid
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_detrended)
ret = o3d.io.write_point_cloud(cloud_detrended, pcd)

#%% Show the clean point cloud

#%% Segment and cluster the point cloud into a point cloud of potential grains

# Find neighbors of point cloud

# Determine node surface

# Compute normals and force them to point towards positive Z

# Initial segmentation

# Cluster label to prevent over-segmentation

# Clean the segmentation
