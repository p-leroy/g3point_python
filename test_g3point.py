import os

import g3point

dir_ = "./data"
cloud = os.path.join(dir_, "Otira_1cm_grains.ply")
ini = os.path.join(dir_, "Otira_1cm_grains.ini")

#%% Create G3Point object
g3point_data = g3point.G3Point(cloud, ini, remove_mins=False)

#%% Initial segmentation
g3point_data.initial_segmentation()

#%% Cluster labels
g3point_data.cluster(version='cpp')

#%% Clean labels
g3point_data.clean(version='cpp')

#%% Save data
out, out_sinks = g3point_data.save()
