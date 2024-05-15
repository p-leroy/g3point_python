import os

import numpy as np

dir_debug = r'C:\DATA\qG3Point\debug'
# np.save(os.path.join(r'C:\DATA\qG3Point\debug', 'py_condition'), condition)

#%% load data
cpp_Aangle = np.genfromtxt(os.path.join(dir_debug, 'A.csv'), delimiter=',')

#%% load python data
py_Aangle = np.load(os.path.join(dir_debug, 'py_Aangle.npy'))

#%% TEST
diff_Aangle = py_Aangle - cpp_Aangle
diff_not_nan = diff_Aangle[np.where(np.isfinite(diff_Aangle))]
max_diff = np.amax(abs(diff_not_nan))
if max_diff < 1e-3:
    print(f'OK matrices in python and in cpp are similar (diff = {max_diff})')
else:
    print(f'WARNING matrices in python and in cpp are different (max_diff = {max_diff})')

#%% load data
cpp_condition = np.genfromtxt(os.path.join(dir_debug, 'symmetrical_condition.csv'), delimiter=',')

#%% load python data
py_condition = np.load(os.path.join(dir_debug, 'py_condition.npy'))

#%% TEST
diff_condition = py_condition - cpp_condition
max_diff = np.amax(abs(diff_condition))
if max_diff < 1e-3:
    print(f'OK matrices in python and in cpp are similar (diff = {max_diff})')
else:
    print(f'WARNING matrices in python and in cpp are different (max_diff = {max_diff})')