import os

import numpy as np
import matplotlib.pyplot as plt

study_site = '1B_debug'

dir_ = 'C:/Users/PaulLeroy/Nextcloud/Partages_recus/paul_leroy'
load_data_folder = os.path.join(dir_, 'Data')
save_results_folder = os.path.join(dir_, 'Results_ple')
save_figure_folder = os.path.join(dir_, 'Figures_ple')

ellipsoid_file = os.path.join(load_data_folder, 'ellipse_fit_' + study_site + '.txt')

#%% Load data
f1 = np.loadtxt(ellipsoid_file, skiprows=1)
n_ellipsoids = len(f1)

#%%
idx_g3point_radius_y = 8
idx_g3point_index = 6

b_axis = 2 * f1[:, idx_g3point_radius_y]  # files are in radius, we need diameters
dx = 1.1 * np.amax(b_axis)
labels_ellipsoid = f1[:, idx_g3point_index]

#########
#########
## ANGLES

#%%
delta = 1e32
idx_g3point_r00 = 10
idx_g3point_r10 = idx_g3point_r00 + 3
idx_g3point_r20 = idx_g3point_r00 + 6

#%%
alpha = np.zeros(n_ellipsoids)
alpha2 = np.zeros(n_ellipsoids)

for k in range(n_ellipsoids):

    p2 = np.array((f1[k, idx_g3point_r00],
                   f1[k, idx_g3point_r10],
                   f1[k, idx_g3point_r20]))  # g3point_r00 g3point_r10 g3point_r20

    p1 = np.array([f1[k, 0], f1[k, 1] + delta, f1[k, 2]])  # x-y plot - mapview (angle with y axis)
    angle = np.arctan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)
    u, v, _ = p2
    if angle > np.pi / 2 or angle < -np.pi / 2:
        u = -u
        v = -v
    alpha[k] = np.arctan(v / u) + np.pi / 2

    p1 = np.array([f1[k, 0], f1[k, 1], f1[k, 2] + delta])  # x-z plot
    angle = np.arctan2(np.linalg.norm(np.cross(p1, p2)), p1 @ p2)
    _, v, w = p2
    if angle > np.pi / 2 or angle < -np.pi / 2:
        v = -v
        w = -w
    alpha2[k] = np.arctan(v / w) + np.pi / 2

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
