import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ScalarFormatter
import matplotlib.ticker as tck

study_site = '1B'

dir_ = 'C:/Users/PaulLeroy/Nextcloud/Partages_recus/paul_leroy'
load_data_folder = os.path.join(dir_, 'Data')
save_results_folder = os.path.join(dir_, 'Results_ple')
save_figure_folder = os.path.join(dir_, 'Figures_ple')

ellipsoid_file = os.path.join(load_data_folder, 'ellipse_fit_' + study_site + '.txt')
pc_file = os.path.join(load_data_folder, 'pc_labelled_' + study_site + '.txt')

n_iter = 10
quantile_args_matlab = {'q': [0.1, 0.5 ,0.9], 'method': 'hazen'}
quantile_args_defaults = {'q': [0.1, 0.5 ,0.9]}

quantile_args = quantile_args_defaults

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

#%%
d = []
r_cpp = np.array([ 0.0348951, 0.500107,
0.427364, 0.904179,
0.285421, 0.462301,
0.292024, 0.765092,
0.988218, 0.308724,
0.0842801, 0.427416,
0.522445, 0.48388,
0.945402, 0.8786,
0.330725, 0.377017,
0.120838, 0.962297]).reshape(-1, 2)

for i in range(n_iter):
    # r = np.random.rand(1, 2)[0]
    r = r_cpp[i, :]
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
d = []

r = np.array([ 0.00125126, 0.563585])
print(r)
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

#%%
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

#%%
my_quantile(d[0], 0.1)

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
# ax.errorbar(dq_final, [0.1, 0.5, 0.9], [0, 0, 0], edq)
ax.set_xlim(min_d, max_d)
ax.set_ylim(0, 1)
ax.set_xlabel('Diameter [mm]')
ax.set_ylabel('CDF')
ax.grid()
ax.xaxis.set_major_formatter(ScalarFormatter())

#%%
delta = 1e32
data = f1

#%%
k = 0
n_ellipsoids = len(f1)
u = np.zeros(n_ellipsoids)
v = np.zeros(n_ellipsoids)
w = np.zeros(n_ellipsoids)
alpha = np.zeros(n_ellipsoids)
for k in range(n_ellipsoids):
    sensor_center = np.array((f1[k, 0], f1[k, 1] + delta, f1[k, 2]))
    # x-y plot - mapview (angle with y axis)
    u[k] = data[k, 10]
    v[k] = data[k, 11]
    w[k] = data[k, 12]

    p1 = sensor_center
    p2 = np.array((u[k], v[k], w[k]))
    angle = np.atan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)

    if angle > np.pi / 2 or angle < -np.pi / 2:
        u[k] = -u[k]
        v[k] = -v[k]
        w[k] = -w[k]

    alpha[k] = np.atan(v[k] / u[k]) + np.pi / 2

granulo_angle_m_view = alpha

#%%
alpha = np.zeros(n_ellipsoids)

for k in range(n_ellipsoids):
    sensor_center = np.array([f1[k, 0], f1[k, 1] + delta, f1[k, 2]])
    # x-y plot - mapview (angle with y axis)
    u, v, w = data[k, 10:13]

    p1 = sensor_center
    p2 = np.array((u, v, w))
    angle = np.atan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)

    if angle > np.pi / 2 or angle < -np.pi / 2:
        u = -u
        v = -v
        w = -w

    alpha[k] = np.atan(v / u) + np.pi / 2

granulo_angle_m_view = alpha * 180 / np.pi

#%%
alpha2 = np.zeros(n_ellipsoids)

for k in range(n_ellipsoids):
    sensor_center = np.array([f1[k, 0], f1[k, 1], f1[k, 2] + delta])
    # x-z plot
    u, v, w = data[k, 10:13]

    p1 = sensor_center
    p2 = np.array((u, v, w))
    angle = np.atan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)
    print(angle)
    if angle > np.pi / 2 or angle < -np.pi / 2:
        u = -u
        v = -v
        w = -w

    alpha2[k] = np.atan(v / w) + np.pi / 2

granulo_angle_x_view = alpha2 * 180 / np.pi

#%%
alpha = np.zeros(n_ellipsoids)
alpha2 = np.zeros(n_ellipsoids)

for k in range(n_ellipsoids):
    u, v, w = data[k, 10:13]
    p2 = np.array((u, v, w))

    p1 = np.array([f1[k, 0], f1[k, 1] + delta, f1[k, 2]])  # x-y plot - mapview (angle with y axis)
    angle = np.atan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)
    u, v, _ = p2
    if angle > np.pi / 2 or angle < -np.pi / 2:
        u = -u
        v = -v
    alpha[k] = np.atan(v / u) + np.pi / 2

    p1 = np.array([f1[k, 0], f1[k, 1], f1[k, 2] + delta])  # x-z plot
    angle = np.atan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)
    _, v, w = p2
    if angle > np.pi / 2 or angle < -np.pi / 2:
        v = -v
        w = -w
    alpha2[k] = np.atan(v / w) + np.pi / 2

granulo_angle_m_view = alpha * 180 / np.pi
granulo_angle_x_view = alpha2 * 180 / np.pi

#%%
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.hist(granulo_angle_m_view, edgecolor='black')
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.set_xlabel('Azimut [°]')

ax2.hist(granulo_angle_x_view, edgecolor='black')
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.set_xlabel('Dip [°]')

fig.tight_layout()