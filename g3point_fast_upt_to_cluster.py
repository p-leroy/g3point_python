import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

import laspy

from g3point_python import tools
from g3point_python.detrend import rotate_point_cloud_plane, orient_normals
from g3point_python.cluster import clean_labels, cluster_labels
from g3point_python.segment import segment_labels

# Inputs
dir_ = r"C:\DATA\PhilippeSteer\G3Point"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
cloud_detrended = os.path.join(dir_, "Otira_1cm_grains_rotated_detrended.ply")
cloud_ardeche = os.path.join(dir_, "Ardeche_2021_inter_survey_C2.part.laz")
ini = r"C:\dev\python\g3point_python\params.ini"

# Load data
xyz = tools.load_data(cloud)
# Remove min values
mins = np.amin(xyz, axis=0)
xyz = xyz - mins

params = tools.read_parameters(ini)

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
[labels, nlabels, stacks, sink_indexes] = cluster_labels(xyz, params, neighbors_indexes, labels, stacks, ndon,
                                                         sink_indexes, surface, normals, v2=True)

#%%
root, ext = os.path.splitext(cloud)
filename = root + '_G3POINT.laz'

las = laspy.create(point_format=7, file_version='1.4')

las.x = xyz[:, 0] + mins[0]
las.y = xyz[:, 1] + mins[1]
las.z = xyz[:, 2] + mins[2]

# set random colors
rng = np.random.default_rng(42)
rgb = rng.random((len(stacks), 3))[labels, :] * 255
las.red = rgb[:, 0]
las.green = rgb[:, 1]
las.blue = rgb[:, 2]

las.add_extra_dim(laspy.ExtraBytesParams(
    name="g3point_label",
    type=np.uint32
))

las.g3point_label = labels

print(f"save {filename}")
las.write(filename)

#%% Save sinks
root, ext = os.path.splitext(cloud)
filename = root + '_G3POINT_SINKS.laz'

las = laspy.create(point_format=7, file_version='1.4')

las.x = xyz[sink_indexes, 0] + mins[0]
las.y = xyz[sink_indexes, 1] + mins[1]
las.z = xyz[sink_indexes, 2] + mins[2]

# set random colors
rng = np.random.default_rng(42)
rgb = rng.random((len(stacks), 3)) * 255
las.red = rgb[:, 0]
las.green = rgb[:, 1]
las.blue = rgb[:, 2]

las.add_extra_dim(laspy.ExtraBytesParams(
    name="g3point_label",
    type=np.uint32
))

las.g3point_label = np.arange(len(stacks))

print(f"save {filename}")
las.write(filename)