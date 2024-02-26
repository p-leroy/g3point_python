import configparser
import os
import random
import colorsys

import laspy
import numpy as np
import open3d as o3d


def get_random_colors_cc(number_of_colors):
    rng = np.random.default_rng(42)
    rg = rng.random((number_of_colors, 2)) * 255
    b = 255 - (rg[:, 0] + rg[:, 1]) / 2
    rgb = np.c_[rg, b]
    return rgb


def get_random_colors(number_of_colors):
    rgb = np.zeros((number_of_colors, 3))
    random.seed(42)
    for k in range(number_of_colors):
        hue, saturation, lightness = random.random(), 0.8 + random.random() / 5.0, 0.5 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(hue, lightness, saturation)]
        rgb[k, :] = r, g, b

    return rgb


def save_data_with_colors(cloud, xyz, mins, stacks, labels, tag):
    head, tail = os.path.split(cloud)
    root, ext = os.path.splitext(tail)
    filename = os.path.join(head, root + tag + '.laz')

    las = laspy.create(point_format=7, file_version='1.4')

    las.x = xyz[:, 0] + mins[0]
    las.y = xyz[:, 1] + mins[1]
    las.z = xyz[:, 2] + mins[2]

    # set random colors
    # rng = np.random.default_rng(42)
    # rgb = rng.random((len(stacks), 3))[labels, :] * 255
    rgb = get_random_colors(len(stacks))[labels, :]
    las.red = rgb[:, 0]
    las.green = rgb[:, 1]
    las.blue = rgb[:, 2]

    las.add_extra_dim(laspy.ExtraBytesParams(
        name="g3point_label",
        type=np.uint32
    ))

    las.g3point_label = labels

    print(f"save {filename}")
    las.write(filename)


def load_data(file):
    ext = os.path.splitext(file)[-1]
    if ext == '.ply':
        pcd_orig = o3d.io.read_point_cloud(file).points
        xyz = np.asarray(pcd_orig)
    elif ext == '.laz':
        las_data = laspy.read(file)
        xyz = np.c_[las_data.X, las_data.Y, las_data.Z]
    else:
        raise TypeError('unhandled extension ' + ext)

    return xyz


class Parameters:
    def __init__(self, ini):
        config = configparser.ConfigParser()
        config.read(ini)
        params = config['DEFAULT']
        self.name = params['name']
        self.iplot = params.getint('iplot')
        self.saveplot = params.getint('saveplot')
        self.denoise = params.getint('denoise')
        self.decimate = params.getint('decimate')
        self.minima = params.getint('minima')
        self.rot_detrend = params.getint('rot_detrend')
        self.clean = params.getint('clean')
        self.grid_by_number = params.getint('grid_by_number')
        self.save_granulo = params.getint('save_granulo')
        self.save_grain = params.getint('save_grain')
        self.res = params.getfloat('res')
        self.n_scale = params.getint('n_scale')
        self.min_scale = params.getfloat('min_scale')
        self.max_scale = params.getfloat('max_scale')
        self.knn = params.getint('knn')
        self.rad_factor = params.getfloat('rad_factor')
        self.max_angle1 = params.getfloat('max_angle1')
        self.max_angle2 = params.getfloat('max_angle2')
        self.min_flatness = params.getfloat('min_flatness')
        self.fit_method = params['fit_method']
        self.a_quality_thresh = params.getfloat('a_quality_thresh')
        self.min_diam = params.getfloat('min_diam')
        self.n_axis = params.getint('n_axis')
        self.n_min = params.getint('n_min')
        self.dx_gbn = params.getfloat('dx_gbn')


def read_parameters(ini):
    params = Parameters(ini)
    return params
