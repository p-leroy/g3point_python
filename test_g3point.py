import os

import g3point

dir_ = "./data"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
ini = os.path.join(dir_, "Otira_1cm_grains.ini")

#%% Create G3Point object
g3point_data = g3point.G3Point(cloud, ini, remove_mins=True)

#%% Initial segmentation
g3point_data.initial_segmentation()

#%% Cluster labels
# version: 'matlab' 'matlab_dbscan' 'cpp' 'custom':
# condition_flag: 'lower' 'upper' 'symmetrical_large' 'symmetrical_strict'
g3point_data.cluster(version='matlab_dbscan')  # 'matlab' 'matlab_dbscan' 'cpp' 'custom'

#%% Clean labels
g3point_data.clean(version='matlab')

#%% Fit ALL ellipsoids
g3point_data.fit_ellipsoids()

#%% Fit ONE ellipsoid
g3point_data.fit_ellipsoid(0)

#%% Save data
out, out_sinks = g3point_data.save()

#%% Plot
