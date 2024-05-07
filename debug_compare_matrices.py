import os

import numpy as np

dir_debug = r'C:\dev\python\g3point_python\data\debug'

#%% load data
cpp_Aangle = np.genfromtxt(os.path.join(dir_debug, 'A.csv'), delimiter=',')

#%% load python data
py_Aangle = np.load(os.path.join(dir_debug, 'py_Aangle.data.npy'))

#%% TEST
diff_Aangle = py_Aangle - cpp_Aangle
diff_not_nan = diff_Aangle[np.where(np.isfinite(diff_Aangle))]
if np.amax(diff_not_nan) < 1e-3:
    print('OK Aangle in python and in cpp are similar')
