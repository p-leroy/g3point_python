import os

import numpy as np

study_site = '1B'

dir_ = 'C:/Users/PaulLeroy/Nextcloud/Partages_recus/paul_leroy'
load_data_folder = os.path.join(dir_, 'Data')
save_results_folder = os.path.join(dir_, 'Results_ple')
save_figure_folder = os.path.join(dir_, 'Figures_ple')

ellipsoid_file = os.path.join(load_data_folder, 'ellipse_fit_' + study_site + '.txt')
pc_file = os.path.join(load_data_folder, 'pc_labelled_' + study_site + '.txt')

n_iter = 100

#%% Load data
f1 = np.loadtxt(ellipsoid_file, skiprows=1)
f2 = np.loadtxt(pc_file, skiprows=1)

#%%
idx_g3point_radius_y = 8
idx_g3point_index = 6

b_axis = 2 * f1[:, idx_g3point_radius_y]  # files are in radius, we need diameters
dx = 1.1 * np.amax(b_axis)
labels_ellipsoid = f1[:, idx_g3point_index]

idx_label_grains = 7
x, y, z = f2[:, :3].T
label_grains = f2[:, idx_label_grains]

#%%
for i in range(n_iter):
    r = np.random.rand(1, 2)[0]
    x_grid = np.arange(np.amin(x) - r[0] * dx, np.amax(x), dx)
    y_grid = np.arange(np.amin(y) - r[1] * dx, np.amax(y), dx)

