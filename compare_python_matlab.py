import os

import open3d as o3d

import g3point
from g3point.visualization import show_clouds

dir_ = "./data"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
ini = os.path.join(dir_, "Otira_1cm_grains.ini")

# WARNING
# When finding sink nodes (local maximum), be careful to the use >= and not >
# matlab_dbscan is the good option to compare between python and matlab

#%% Create G3Point object
g3point_data = g3point.G3Point(cloud, ini, remove_mins=True)

#%% Initial segmentation
g3point_data.initial_segmentation()

#%% Cluster labels
# WARNING with the Mmerge matrix copied from matlab, the clustering by DBSCAN is slightly different between matlab and
# python
# CHECK Aangle is slightly different between matlab and python, giving a slightly different condition matrix
g3point_data.cluster(version='matlab_dbscan')

#%%
pcd, pcd_sinks = g3point_data.get_pcd_and_pcd_sinks()

clouds = (
    ('pcd', pcd, None, 3),  # name / cloud / color / size
    ('pcd_sinks', pcd_sinks, None, 5),
)
show_clouds(clouds, warnings=True, projection='orthographic')

#%% COMPARE DBSCAN BETWEEN MATLAB AND PYTHON
import pickle

import numpy as np
from scipy.io import loadmat
mat = loadmat(r"C:\dev\python\g3point_python\data\Mmerge.mat")
Mmerge_matlab = mat['Mmerge']
Mmerge_matlab[Mmerge_matlab == np.inf] = 1e9
with open('c:/dev/python/g3point_python/data/Mmerge.pkl', 'rb') as f:
    Mmerge_python = pickle.load(f)
idx = (Mmerge_matlab != Mmerge_python)