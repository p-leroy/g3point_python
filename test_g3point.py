import os

import g3point

dir_ = "./data"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
ini = os.path.join(dir_, "Otira_1cm_grains.ini")

#%% Create G3Point object
g3point_data = g3point.G3Point(cloud, ini, remove_mins=True)

#%% Initial segmentation
g3point_data.initial_segmentation()

#%% Fit ellipsoids
g3point_data.fit_ellipsoids()

#%% Cluster labels
g3point_data.cluster(version='cpp')

#%% Clean labels
g3point_data.clean(version='cpp')

#%% Save data
out, out_sinks = g3point_data.save()

#%% Try some fitting
xyz_grain = g3point_data.xyz[g3point_data.stacks[0], :]
center, radii, quaternions, rotation_matrix, ellipsoid_parameters = g3point.fit_ellipsoid_to_grain(xyz_grain)

#%% Plot
