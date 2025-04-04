import os

import scipy.io

from g3point.ellipsoid import inertia_fit

# from .ellipsoid import inertia_fit

dir_ = r'C:\dev\python\g3point_python\data'

#%%
mat = scipy.io.loadmat(os.path.join(dir_, 'xyz_for_inertia_fit_test.mat'))

#%%
R, center, radii = inertia_fit(mat['xyz'])
