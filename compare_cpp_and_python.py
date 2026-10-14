import os
from pathlib import Path

import numpy as np

from lidar_platform import sbf

# To compare with the C++ code:
# Compile CloudCompare with the option USE_OPEN3D_WITH_G3POINT
# Export matrices from C++ using EigenArrayToFile
# In python, use the version='cpp' parameter

dir_debug = os.path.join(Path.home(), 'g3point')

#%% load data
cpp_Aangle = np.genfromtxt(os.path.join(dir_debug, 'Aangle.csv'), delimiter=',')

#%% load python data
py_Aangle = np.load(os.path.join(dir_debug, 'Aangle.npy'))

#%% TEST
diff_Aangle = py_Aangle - cpp_Aangle
diff_not_nan = diff_Aangle[np.where(np.isfinite(diff_Aangle))]
max_diff = np.amax(abs(diff_not_nan))
if max_diff < 1e-3:
    print(f'OK matrices in python and in cpp are similar (diff = {max_diff})')
else:
    print(f'WARNING matrices in python and in cpp are different (max_diff = {max_diff})')

#%% load data
cpp_condition = np.genfromtxt(os.path.join(dir_debug, 'condition.csv'), delimiter=',')

#%% load python data
py_condition = np.load(os.path.join(dir_debug, 'condition.npy'))

#%% TEST
diff_condition = py_condition - cpp_condition
max_diff = np.amax(abs(diff_condition))
if max_diff < 1e-3:
    print(f'OK matrices in python and in cpp are similar (diff = {max_diff})')
else:
    print(f'WARNING matrices in python and in cpp are different (max_diff = {max_diff})')

# %% COMPARE THE RESULTING LABELS
py_labels = np.load(os.path.join(dir_debug, 'labels.npy'))
cc_labels = sbf.read(os.path.join(dir_debug, 'Otira_1cm_grains - Cloud.sbf')).g3point_label

diff_labels = py_labels - cc_labels
if (len(unique := np.unique(diff_labels))) == 1:
    if unique[0] == 0:
        print("All labels are the same")
    else:
        print(f"Unique value unexpected: {unique}")
else:
    print(f'WARNING labels are not unique (unique = {unique})')