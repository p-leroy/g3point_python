import numpy as np


def fit_plane(xyz):
    # Fit a plane of coordinates z = ax + by + c using a singular value decomposition
    # see plane_fit by Kevin Moerman
    # https://fr.mathworks.com/matlabcentral/fileexchange/22042-plane_fit
    centroid = np.mean(xyz, axis=0)
    xyz_c = xyz - centroid  # centered coordinates
    U, S, Vh = np.linalg.svd(xyz_c, full_matrices=False)  # singular value decomposition
    n = -Vh.T[:, 2] /  Vh[-1, -1]  # careful, Vh is already transposed, this is different from Matlab
    a = n[0]
    b = n[1]
    c = -np.dot(centroid, n)
    x, y, z = np.hsplit(xyz, 3)
    dist_signed = (c + a * x + b * y - z) / (a ** 2 + b ** 2 + 1) ** 0.5
    return a, b, c, dist_signed


def orient_normal(scene_center, normal, sensor_center):
    # Flip the normals, so they are oriented towards the sensor center
    # x,y,z: scene_center
    # u,v,w: normals
    # ox,oy,oz: sensor center

    p1 = sensor_center - scene_center
    p2 = normal

    # Flip the normal vector if it is not pointing towards the sensor
    angle = np.arctan2(np.linalg.norm(np.cross(p1, p2)), np.dot(p1, p2))
    if angle > np.pi/2 or angle < -np.pi/2:
        normal = -normal  # invert normal
    return normal


def orient_normals(points, normals, sensor_center):
    # Flip the normals, so they are oriented towards the sensor center
    # x,y,z: points
    # u,v,w: normals
    # ox,oy,oz: sensor center

    p1 = sensor_center - points
    p2 = normals

    # Flip the normals if they are not pointing towards the sensor
    angle = np.arctan2(np.linalg.norm(np.cross(p1, p2), axis=1), np.sum(p1 * p2, axis=1))
    index = (angle > np.pi / 2) | (angle < -np.pi / 2)
    normals[index] = -normals[index]  # invert normal

    return normals


def get_skew_symmetric_cross_prduct_matrix(v):

    skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]])

    return skew


def vec2rot(a, b, method='Rik'):
    # from "Calculate Rotation Matrix to align Vector A to Vector Bin 3D?" on math.stackexchange.com
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/180436#180436

    # normalize a and b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    s = np.linalg.norm(v)  # sine of angle
    c = np.dot(a, b)  # cosine of angle

    if method == 'Rik':
        skew = get_skew_symmetric_cross_prduct_matrix(v)
        rotation = np.eye(3) + skew + skew @ skew * (1 - c) / s**2

    if method == 'Kjetil':  # a and be have to be normalized
        G = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        Fi = np.c_[a, (b - c * a) / np.linalg.norm(b - c * a), np.cross(b,a)]
        rotation = Fi @ G @ np.linalg.inv(Fi)

    return rotation


def rotate_point_cloud_plane(xyz, axis):

    point0 = np.array([0, 0, 0])
    sensor_center = np.array([0, 0, 1e32])

    a, b, c, dist_signed = fit_plane(xyz)  # fit a plane to the data (z = ax + by + c)
    normal = np.array([-a, -b, 1])  # get a normal
    normal = orient_normal(point0, normal, sensor_center)  # make the normal point along the z direction
    rotation = vec2rot(normal, axis)  # compute a rotation to align the normal with the axis
    centroid = np.mean(xyz, axis=0)
    xyz_detrended = (rotation @ (xyz - centroid).T).T + centroid  # detrend the coordinates

    return xyz_detrended
