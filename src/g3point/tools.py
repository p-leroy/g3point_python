import os
import random
import colorsys

import laspy
import numpy as np
import open3d as o3d


def get_random_colors(number_of_colors, version=None):

    if version == 'hsv':
        rgb = np.zeros((number_of_colors, 3))
        random.seed(42)
        for k in range(number_of_colors):
            hue, saturation, lightness = random.random(), 0.8 + random.random() / 5.0, 0.5 + random.random() / 5.0
            r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(hue, lightness, saturation)]
            rgb[k, :] = r, g, b

    elif version == 'cc':
        rng = np.random.default_rng(42)
        rg = rng.random((number_of_colors, 2)) * 255
        b = 255 - (rg[:, 0] + rg[:, 1]) / 2
        rgb = np.c_[rg, b]

    else:
        rng = np.random.default_rng(42)
        rgb = rng.random((number_of_colors, 3)) * (2**16 - 1)

    return rgb


def save_data_with_colors(cloud, xyz, stacks, labels, tag):
    head, tail = os.path.split(cloud)
    root, ext = os.path.splitext(tail)
    filename = os.path.join(head, root + tag + '.laz')

    # scale data to avoid loss of accuracy with LAS format
    x, y, z = np.split(xyz, 3, axis=1)

    # 1. Create a new header
    header = laspy.LasHeader(point_format=7, version="1.4")

    # 2. Create a Las
    las = laspy.LasData(header)

    las.x = np.squeeze(x)
    las.y = np.squeeze(y)
    las.z = np.squeeze(z)

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

    return filename


def load_data(file, dtype=None):
    ext = os.path.splitext(file)[-1]
    if ext == '.ply':
        pcd_orig = o3d.io.read_point_cloud(file).points
        xyz = np.asarray(pcd_orig)
    elif ext == '.laz':
        las_data = laspy.read(file)
        xyz = np.c_[las_data.X, las_data.Y, las_data.Z]
    else:
        raise TypeError('unhandled extension ' + ext)

    if dtype is not None:
        xyz = xyz.astype(dtype)

    return xyz


def check_stacks(stacks, number_of_points):

    # Initialize the set of indexes with the first stack
    stack = stacks[0]
    myset = {*stack}

    # Initialize min and max
    min = float('inf')
    max = float('-inf')

    for idx in stack:
        if idx < min:
            min = idx
        if idx > max:
            max = idx

    for stack in stacks[1:]:
        for idx in stack:
            if idx < min:
                min = idx
            if idx > max:
                max = idx
        myset.update(stack)

    # Check the coherency of the stack
    if len(myset) != number_of_points:  # number of values in the set
        raise ValueError('stacks are not coherent: the length of the set shall be equal to the number of points')
    if min != 0:  # min value in the set
        raise ValueError('stacks are not coherent: min shall be 0')

    return True