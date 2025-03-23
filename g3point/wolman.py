import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ScalarFormatter

study_site = '1B_debug'

dir_ = 'C:/Users/PaulLeroy/Nextcloud/Partages_recus/paul_leroy'
load_data_folder = os.path.join(dir_, 'Data')
save_results_folder = os.path.join(dir_, 'Results_ple')
save_figure_folder = os.path.join(dir_, 'Figures_ple')

ellipsoid_file = os.path.join(load_data_folder, 'ellipse_fit_' + study_site + '.txt')
pc_file = os.path.join(load_data_folder, 'pc_labelled_' + study_site + '.txt')

n_iter = 3
quantile_args_matlab = {'q': [0.1, 0.5 ,0.9], 'method': 'hazen'}
quantile_args_defaults = {'q': [0.1, 0.5 ,0.9]}

quantile_args = quantile_args_defaults

def my_quantile(x, q):  # same results as Python with the default parameters
    if q > 1.0 or q < 0:
        raise ValueError('q must be between 0 and 1')
    n = len(x)
    x = np.sort(x)
    id = (n - 1) * q
    lo = int(np.floor(id))
    hi = int(np.ceil(id))
    qs = x[lo]
    h = (id - lo)

    return (1.0 - h) * qs + h * x[hi]

#%% Load data
f1 = np.loadtxt(ellipsoid_file, skiprows=1)
f2 = np.loadtxt(pc_file, skiprows=1)
n_ellipsoids = len(f1)

#%%
idx_g3point_radius_y = 8
idx_g3point_index = 6

b_axis = 2 * f1[:, idx_g3point_radius_y]  # files are in radius, we need diameters
dx = 1.1 * np.amax(b_axis)
labels_ellipsoid = f1[:, idx_g3point_index]

idx_label_grains = 7
x, y, z = f2[:, :3].T
labels_grains = f2[:, idx_label_grains]

#########
#########
## WOLMAN

#%%
d = []

r_array = np.array([0.617071, 0.908485,
0.0636531, 0.983472,
0.630667, 0.956718,]).reshape(n_iter, -1)

for i in range(n_iter):
    # r = np.random.rand(1, 2)[0]
    r = r_array[i, :]
    print(i, r)
    x_grid = np.arange(np.amin(x) - r[0] * dx, np.amax(x), dx)
    y_grid = np.arange(np.amin(y) - r[1] * dx, np.amax(y), dx)
    n_x = len(x_grid)
    n_y = len(y_grid)
    dist = np.zeros((n_x, n_y))
    i_wolman = np.zeros((n_x, n_y))
    for ix in range(n_x):
        for iy in range(n_y):
            distances = ((x - x_grid[ix]) ** 2 + (y - y_grid[iy]) ** 2) ** 0.5
            idx = np.argmin(distances)
            dist[ix, iy] = distances[idx]
            i_wolman[ix, iy] = idx
    i_wolman_selection = i_wolman[dist < dx / 10].astype(int)
    wolman_selection = labels_grains[i_wolman_selection]
    _, _, y_ind = np.intersect1d(wolman_selection, labels_ellipsoid,
                                 return_indices=True)
    d.append(b_axis[y_ind] * 1000)  # conversion to millimeters

#%%
dq = np.empty((n_iter, 3))
d_sample = d[0]
dq[0, :] = np.quantile(d[0], **quantile_args)
for i in range(1, n_iter):
    d_sample = np.r_[d_sample, d[i]]
    dq[i, :] = np.quantile(d[i], **quantile_args)

#%%
edq = np.std(dq, axis=0, ddof=1)
dq_final = np.quantile(d_sample, **quantile_args)

#%% Save results in a text file
import csv
with open(os.path.join(save_results_folder, study_site + '_GSD_G3POINT.txt'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE, escapechar=' ')
    writer.writerow(['#', 'D10', 'D50', 'D90', '[mm]'])
    writer.writerow(dq_final)
    writer.writerow(['#', 'std(D10)', 'std(D50)', 'std(D90)', '[mm]'])
    writer.writerow(edq)

#%%
def format_func(value, tick_number):
    return '%g' % (value)


n_samples = len(d_sample)
min_d = np.amin(d_sample)
max_d = np.amax(d_sample)
fig, ax = plt.subplots(1, 1)
ax.semilogx(np.sort(d_sample), np.arange(n_samples) / n_samples)
ax.errorbar(dq_final, [0.1, 0.5, 0.9], [0, 0, 0], edq)
ax.set_xlim(min_d, max_d)
ax.set_ylim(0, 1)
ax.set_xlabel('Diameter [mm]')
ax.set_ylabel('CDF')
ax.grid()
ax.xaxis.set_major_formatter(ScalarFormatter())
