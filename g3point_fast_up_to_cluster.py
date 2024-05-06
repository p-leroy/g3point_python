import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from g3point_python import tools
from tools.detrend import rotate_point_cloud_plane, orient_normals
from tools.segment import segment_labels

from tools import ellipsoid, Parameters

# Inputs
dir_ = r"C:\DATA\PhilippeSteer\G3Point"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
cloud_test_laz = os.path.join(dir_, "test.laz")
cloud_detrended = os.path.join(dir_, "Otira_1cm_grains_rotated_detrended.ply")
cloud_ardeche = os.path.join(dir_, "Ardeche_2021_inter_survey_C2.part.laz")
ini = r"C:\dev\python\g3point_python\params.ini"

# Load data
xyz = tools.load_data(cloud)
# Remove min values
if True:
    mins = np.amin(xyz, axis=0)  # WARNING: done in the Matlab code
    xyz = xyz - mins

params = Parameters.Parameters(ini)

# Rotate and detrend the point cloud
if params.rot_detrend:
    # make the normal of the fitted plane aligned with the axis
    axis = np.array([0, 0, 1])
    xyz_rot = rotate_point_cloud_plane(xyz, axis)
    x, y, z = np.split(xyz_rot, 3, axis=1)
    # Remove the polynomial trend from the cloud
    least_squares_solution, residuals, rank, s = np.linalg.lstsq(np.c_[np.ones(x.shape), x, y, x ** 2, x * y, y ** 2],
                                                                 z, rcond=None)
    a0, a1, a2, a3, a4, a5 = least_squares_solution
    z_detrended = z - (a0 + a1 * x + a2 * y + a3 * x ** 2 + a4 * x * y + a5 * y ** 2)
    xyz_detrended = np.c_[x, y, z_detrended]
else:
    xyz_detrended = xyz

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Find neighbors of point cloud
tree = KDTree(xyz)  # build a KD tree
neighbors_distances, neighbors_indexes = tree.query(xyz, params.knn + 1)
neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]

# Determine node surface
surface = np.pi * np.amin(neighbors_distances, axis=1) ** 2

# Compute normals and force them to point towards positive Z
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(params.knn))
centroid = np.mean(xyz, axis=0)
sensor_center = np.array([centroid[0], centroid[1], 1000])
normals = orient_normals(xyz, np.asarray(pcd.normals), sensor_center)

# Initial segmentation
labels, nlabels, labelsnpoint, stacks, ndon, sink_indexes = segment_labels(xyz_detrended, params.knn, neighbors_indexes)

# Cluster labels
# [labels, nlabels, stacks, sink_indexes] = cluster(xyz, params, neighbors_indexes, labels, stacks, ndon,
#                                                   sink_indexes, surface, normals,
#                                                   version='cpp', condition_flag=None)

# Clean labels
# [labels, nlabels, stacks, sink_indexes] = clean_labels(xyz, params, neighbors_indexes, labels, stacks, ndon, normals,
#                                                        version='cpp', condition_flag='symmetrical_strict')

#%%
# do not forget to add mins if needed
g3point = tools.save_data_with_colors(cloud, xyz + mins,
                                  stacks, labels, '_G3POINT')
g3point_sinks = tools.save_data_with_colors(cloud, xyz[sink_indexes, :] + mins,
                                  stacks, np.arange(len(stacks)), '_G3POINT_SINKS')

#%% show initial segmentation
colors = np.random.rand(len(stacks), 3)[labels, :]
pcd.colors = o3d.utility.Vector3dVector(colors)
# build pcd_sinks
pcd_sinks = o3d.geometry.PointCloud()
pcd_sinks.points = o3d.utility.Vector3dVector(xyz_detrended[sink_indexes, :])
pcd_sinks.paint_uniform_color(np.array([1., 0., 0.]))

#%%
# coeff4 = 1 / 0.6608698784014869
center, radii, quaternions, rotation_matrix, ellipsoid_parameters = (
    ellipsoid.fit_ellipsoid_to_grain(xyz[stacks[351]]))

#%% Build an ellipsoid as a cloud
xx, yy, zz = ellipsoid.ellipsoid(0, 0, 0, radii[0], radii[1], radii[2])
xyz_grain = np.c_[xx.flatten(), yy.flatten(), zz.flatten()] @ rotation_matrix + center.reshape(1, -1)

grain = o3d.geometry.PointCloud()
grain.points = o3d.utility.Vector3dVector(xyz_grain)
grain.paint_uniform_color((0, 0, 0))
grain.estimate_normals()
grain.orient_normals_consistent_tangent_plane(1)

mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(grain, depth=9)
# radii_ball = [0.1, 0.2]
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(grain, o3d.utility.DoubleVector(radii))
# alpha = 1
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(grain, alpha)
mesh.paint_uniform_color((1, 1, 1))
mesh.compute_vertex_normals()

#%%
o3d.visualization.draw([pcd, grain, mesh])
