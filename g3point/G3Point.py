import os

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from .cluster import clean_labels, cluster
from .detrend import orient_normals, rotate_point_cloud_plane
from .G3PointParameters import G3PointParameters
from .segment import segment_labels
from .tools import load_data, save_data_with_colors
from .ellipsoid import fit_ellipsoid_to_grain


class G3Point:
    def __init__(self, cloud, ini, remove_mins=True):

        # test that files exist
        if not os.path.exists(cloud):
            raise FileNotFoundError(cloud)

        if not os.path.exists(ini):
            raise FileNotFoundError(ini)

        self.cloud = cloud
        self.ini = ini

        # Load data
        self.xyz = load_data(self.cloud)
        # Remove min values
        self.remove_mins = remove_mins
        if self.remove_mins:
            print('WARNING original data shifted (minimums removed)')
            self.mins = np.amin(self.xyz, axis=0)  # WARNING: done in the Matlab code
            self.xyz = self.xyz - self.mins

        # Load parameters
        self.params = G3PointParameters(self.ini)

        # Variables which will be set during the initial_segmentation call
        self.initial_labels = None
        self.initial_stacks = None
        self.ndon = None
        self.initial_sink_indexes = None
        self.surface = None
        self.normals = None
        self.neighbors_indexes = None

        # Variables which will be modified during the clustering and / or cleaning
        self.labels = None
        self.stacks = None
        self.sink_indexes = None

        self.g3point_results = None

    def initial_segmentation(self):

        # Rotate and detrend the point cloud

        if self.params.rot_detrend:
            print('WARNING original data will be detrended')
            # make the normal of the fitted plane aligned with the axis
            axis = np.array([0, 0, 1])
            xyz_rot = rotate_point_cloud_plane(self.xyz, axis)
            x, y, z = np.split(xyz_rot, 3, axis=1)
            # Remove the polynomial trend from the cloud
            least_squares_solution, residuals, rank, s = np.linalg.lstsq(
                np.c_[np.ones(x.shape), x, y, x ** 2, x * y, y ** 2],
                z, rcond=None)
            a0, a1, a2, a3, a4, a5 = least_squares_solution
            z_detrended = z - (a0 + a1 * x + a2 * y + a3 * x ** 2 + a4 * x * y + a5 * y ** 2)
            self.xyz = np.c_[x, y, z_detrended]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)

        # Find neighbors of point cloud
        tree = KDTree(self.xyz)  # build a KD tree
        neighbors_distances, neighbors_indexes = tree.query(self.xyz, self.params.knn + 1)
        # remove the first column in the distances and in the indexes
        neighbors_distances, neighbors_indexes = neighbors_distances[:, 1:], neighbors_indexes[:, 1:]
        self.neighbors_indexes = neighbors_indexes

        # Determine node surface
        self.surface = np.pi * np.amin(neighbors_distances, axis=1) ** 2

        # Compute normals and force them to point towards positive Z
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(self.params.knn))
        centroid = np.mean(self.xyz, axis=0)
        sensor_center = np.array([centroid[0], centroid[1], 1000])
        self.normals = orient_normals(self.xyz, np.asarray(pcd.normals), sensor_center)

        # Initial segmentation
        res = segment_labels(self.xyz, self.params.knn, neighbors_indexes)
        self.initial_labels, self.initial_stacks, self.ndon, self.initial_sink_indexes = res

        self.labels = np.copy(self.initial_labels)
        self.stacks = self.initial_stacks.copy()
        self.sink_indexes = np.copy(self.initial_sink_indexes)

    def cluster(self, version='cpp', condition_flag=None):
        """

        :param version: 'matlab' 'matlab_dbscan' 'cpp' 'custom':
        :param condition_flag: 'lower' 'upper' 'symmetrical_large' 'symmetrical_strict'
        :return:
        """

        res = cluster(self.xyz, self.params, self.neighbors_indexes, self.initial_labels, self.initial_stacks,
                      self.ndon, self.initial_sink_indexes, self.surface, self.normals,
                      version=version, condition_flag=condition_flag)

        self.labels, self.stacks, self.sink_indexes = res

    def clean(self, version='cpp', condition_flag=None):
        """
        Default configuration is 'cpp' 'symmetrical_strict'
        'cpp' forces the condition_flag value to 'symmetrical_strict'
        :param version: 'matlab' 'matlab_dbscan' 'cpp' 'custom':
        :param condition_flag: 'lower' 'upper' 'symmetrical_large' 'symmetrical_strict'
        :return:
        """

        res = clean_labels(self.xyz, self.params, self.neighbors_indexes, self.labels, self.stacks,
                           self.ndon, self.normals,
                           version=version, condition_flag=condition_flag)

        self.labels, self.stacks, self.sink_indexes = res

    def save(self):

        if self.remove_mins:
            g3point = save_data_with_colors(self.cloud, self.xyz + self.mins,
                                            self.stacks, self.labels, '_G3POINT')
            g3point_sinks = save_data_with_colors(self.cloud, self.xyz[self.sink_indexes, :] + self.mins,
                                                self.stacks, np.arange(len(self.stacks)), '_G3POINT_SINKS')
        else:
            g3point = save_data_with_colors(self.cloud, self.xyz,
                                            self.stacks, self.labels, '_G3POINT')
            g3point_sinks = save_data_with_colors(self.cloud, self.xyz[self.sink_indexes, :],
                                                  self.stacks, np.arange(len(self.stacks)), '_G3POINT_SINKS')

        return g3point, g3point_sinks

    def fit_ellipsoid(self, label):
        stack = self.stacks[label]
        xyz_grain = self.xyz[stack, :]
        center, radii, quaternions, rotation_matrix, ellipsoid_parameters = fit_ellipsoid_to_grain(xyz_grain)
        return center, radii, quaternions, rotation_matrix, ellipsoid_parameters

    def fit_ellipsoids(self):
        self.g3point_results = np.zeros((len(self.stacks), 3 + 3 + 9))
        for label, stack in enumerate(self.stacks):
            progress = int(label / len(self.stacks) * 100)
            if progress % 10 == 0:
                print(f'progress {progress}%')
            xyz_grain = self.xyz[stack, :]
            center, radii, quaternions, rotation_matrix, ellipsoid_parameters = fit_ellipsoid_to_grain(xyz_grain)
            self.g3point_results[label, 0:3] = center
            self.g3point_results[label, 3:6] = radii
            self.g3point_results[label, 6:15] = rotation_matrix.flatten()
