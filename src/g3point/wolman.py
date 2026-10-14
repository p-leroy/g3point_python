import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ScalarFormatter

study_site = '1B_debug'
study_site = 'Raniskca_2023'

dir_ = 'C:/Users/PaulLeroy/Nextcloud/Partages_recus/paul_leroy'
load_data_folder = os.path.join(dir_, 'Data')
save_results_folder = os.path.join(dir_, 'Results_ple')
save_figure_folder = os.path.join(dir_, 'Figures_ple')

ellipsoid_file = os.path.join(load_data_folder, 'ellipse_fit_' + study_site + '.txt')
pc_file = os.path.join(load_data_folder, 'pc_labelled_' + study_site + '.txt')

n_iter = 100
quantile_args_matlab = {'q': [0.1, 0.5 ,0.9], 'method': 'hazen'}  # to get the same values as in Matlab
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
f1 = np.loadtxt(ellipsoid_file)
f2 = np.loadtxt(pc_file, skiprows=1)
n_ellipsoids = len(f1)

#%%
idx_g3point_radius_y = 8
idx_g3point_index = 6

b_axis = 2 * f1[:, idx_g3point_radius_y]  # files are in radius, we need diameters
dx = 1.1 * np.amax(b_axis)
labels_ellipsoid = f1[:, idx_g3point_index]

if study_site == 'Raniskca_2023':
    idx_label_grains = 6  # check the value of the index, shifts may occur due to additional SF
else:
    idx_label_grains = 7
x, y, z = f2[:, :3].T
labels_grains = f2[:, idx_label_grains]

#########
#########
## WOLMAN

#%%
d = []

r_array = np.array([
0.342763, 0.674834,
0.194105, 0.0343446,
0.735924, 0.920425,
0.374202, 0.307994,
0.194105, 0.0343446,
0.269272, 0.899928,
0.194105, 0.0343446,
0.194105, 0.0343446,
0.0343446, 0.269272,
0.577564, 0.173418,
0.0343446, 0.269272,
0.122769, 0.942243,
0.0435456, 0.619085,
0.0343446, 0.269272,
0.194105, 0.0343446,
0.194105, 0.0343446,
0.269272, 0.899928,
0.269272, 0.899928,
0.377353, 0.476875,
0.626284, 0.409264,
0.650356, 0.523059,
0.945514, 0.372996,
0.429332, 0.465342,
0.28425, 0.389036,
0.431494, 0.276293,
0.355715, 0.546873,
0.443642, 0.0786446,
0.174753, 0.131386,
0.816806, 0.679851,
0.917356, 0.219295,
0.194105, 0.0343446,
0.626702, 0.478632,
0.598272, 0.0236794,
0.519, 0.909665,
0.667826, 0.620633,
0.194105, 0.0343446,
0.0547241, 0.87787,
0.310299, 0.678141,
0.391297, 0.671057,
0.90945, 0.992564,
0.421794, 0.221541,
0.718095, 0.0169171,
0.273634, 0.876473,
0.212281, 0.812932,
0.221835, 0.0481579,
0.309777, 0.724653,
0.31443, 0.018946,
0.877507, 0.990619,
0.0469546, 0.547118,
0.919573, 0.186367,
0.494447, 0.542519,
0.395453, 0.371726,
0.991277, 0.848632,
0.202602, 0.485252,
0.438565, 0.81036,
0.496055, 0.443333,
0.57631, 0.0915517,
0.159192, 0.909537,
0.180493, 0.867223,
0.674517, 0.923777,
0.929845, 0.121734,
0.456801, 0.284111,
0.992504, 0.00452963,
0.00803778, 0.482651,
0.741926, 0.990466,
0.569307, 0.86476,
0.269272, 0.899928,
0.0319336, 0.301942,
0.611362, 0.565328,
0.564228, 0.366008,
0.923075, 0.757768,
0.0249105, 0.836582,
0.190998, 0.108219,
0.269272, 0.899928,
0.979069, 0.574056,
0.440246, 0.938468,
0.937361, 0.327834,
0.790918, 0.481138,
0.897884, 0.33299,
0.0343446, 0.269272,
0.63968, 0.585525,
0.138392, 0.178232,
0.300285, 0.246165,
0.711621, 0.739949,
0.48139, 0.128053,
0.760453, 0.873918,
0.792509, 0.197078,
0.173742, 0.811333,
0.636642, 0.858231,
0.620569, 0.62616,
0.636763, 0.953435,
0.195304, 0.538306,
0.5619, 0.449433,
0.964955, 0.129683,
0.743471, 0.42608,
0.689011, 0.161488,
0.603654, 0.767218,
0.0800864, 0.845364,
0.860211, 0.511092,
0.54791, 0.95669,]).reshape(n_iter, -1)

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
    # _, _, y_ind = np.intersect1d(wolman_selection, labels_ellipsoid,
    #                              return_indices=True)
    y_ind = wolman_selection.astype(int)

    # remove not valid b_axis
    d.append(b_axis[y_ind][b_axis[y_ind] != 0.0] * 1000)  # conversion to millimeters

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
ax.semilogx(np.sort(tmp[:, 0]), np.arange(n_samples) / n_samples)
ax.errorbar(dq_final, [0.1, 0.5, 0.9], [0, 0, 0], edq)
ax.set_xlim(min_d, max_d)
ax.set_ylim(0, 1)
ax.set_xlabel('Diameter [mm]')
ax.set_ylabel('CDF')
ax.grid()
ax.xaxis.set_major_formatter(ScalarFormatter())

#%% READ C++ results
tmp = np.loadtxt(r"C:\Users\PaulLeroy\Nextcloud\Partages_recus\paul_leroy\Data\Raniskca_2023\Wolman.csv",
                 skiprows=5)

#%%
