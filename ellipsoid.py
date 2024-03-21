# This python module is largely inspired by the following work:
#  Levente Hunyadi (2024). Fitting quadratic curves and surfaces
#  (https://www.mathworks.com/matlabcentral/fileexchange/45356-fitting-quadratic-curves-and-surfaces),
#  MATLAB Central File Exchange. Retrieved March 21, 2024.

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import scipy

import numpy as np


def explicit_to_implicit(center, radii, rotation_matrix):
    """
    Cast ellipsoid defined with explicit parameters to implicit vector form.
    Examples:
        p = ellipse_ex2im([xc,yc,zc],[xr,yr,zr],eye(3,3));
    Copyright 2011 Levente Hunyadi
    2024 Paul Leroy for the present Python translation
    :param center: 
    :param radii: 
    :param rotation_matrix: 
    :return: 
    """

    xc, yc, zc = center
    
    xrr, yrr, zrr = 1/radii

    rotation_matrix_flattened = rotation_matrix.flatten('F')
    r11 = rotation_matrix_flattened[0]
    r21 = rotation_matrix_flattened[1]
    r31 = rotation_matrix_flattened[2]
    r12 = rotation_matrix_flattened[3]
    r22 = rotation_matrix_flattened[4]
    r32 = rotation_matrix_flattened[5]
    r13 = rotation_matrix_flattened[6]
    r23 = rotation_matrix_flattened[7]
    r33 = rotation_matrix_flattened[8]

    # terms collected from symbolic expression
    p = np.array([r11**2 * xrr**2 + r21**2 * yrr**2 + r31**2 * zrr**2,
                  r12**2 * xrr**2 + r22**2 * yrr**2 + r32**2 * zrr**2,
                  r13**2 * xrr**2 + r23**2 * yrr**2 + r33**2 * zrr**2,
                  2 * r11 * r12 * xrr**2 + 2 * r21 * r22 * yrr**2 + 2 * r31 * r32 * zrr**2,
                  2 * r11 * r13 * xrr**2 + 2 * r21 * r23 * yrr**2 + 2 * r31 * r33 * zrr**2,
                  2 * r12 * r13 * xrr**2 + 2 * r22 * r23 * yrr**2 + 2 * r32 * r33 * zrr**2,
                  (-2) * (r11**2*xc*xrr**2 + r21**2*xc*yrr**2 + r31**2*xc*zrr**2 + r11*r12*xrr**2*yc + r11 * r13 * xrr**2 * zc + r21 * r22 * yc * yrr**2 + r21 * r23 * yrr**2 * zc + r31 * r32 * yc * zrr**2 + r31 * r33 * zc * zrr**2),
                  (-2) * (r12**2*xrr**2*yc + r22**2*yc*yrr**2 + r32**2*yc*zrr**2 + r11*r12*xc*xrr**2 + r21 * r22 * xc * yrr**2 + r12 * r13 * xrr**2 * zc + r31 * r32 * xc * zrr**2 + r22 * r23 * yrr**2 * zc + r32 * r33 * zc * zrr**2),
                  (-2) * (r13**2*xrr**2*zc + r23**2*yrr**2*zc + r33**2*zc*zrr**2 + r11*r13*xc*xrr**2 + r12 * r13 * xrr**2 * yc + r21 * r23 * xc * yrr**2 + r22 * r23 * yc * yrr**2 + r31 * r33 * xc * zrr**2 + r32 * r33 * yc * zrr**2),
                  r11**2 * xc**2 * xrr**2 + 2 * r11 * r12 * xc * xrr**2 * yc + 2 * r11 * r13 * xc * xrr**2 * zc +
                  r12**2 * xrr**2 * yc**2 + 2 * r12 * r13 * xrr**2 * yc * zc +
                  r13**2 * xrr**2 * zc**2 + r21**2 *xc**2 * yrr**2 +
                  2 * r21 * r22 * xc * yc * yrr**2 + 2 * r21 * r23 * xc * yrr**2 * zc +
                  r22**2 * yc**2 * yrr**2 + 2 * r22 * r23 * yc * yrr**2 * zc +
                  r23**2 * yrr**2 * zc**2 + r31**2 * xc**2 * zrr**2 + 2 * r31 * r32 * xc * yc * zrr**2 +
                  2 * r31 * r33 * xc * zc * zrr**2 + r32**2 * yc**2 * zrr**2 + 2 * r32 * r33 * yc * zc * zrr**2 +
                  r33**2 * zc**2 * zrr**2 - 1])

    return p


def implicit_to_explicit(p):
    """
    Cast ellipsoid defined with implicit parameter vector to explicit form.
    The implicit equation of a general ellipse is
    F(x,y,z) = Ax**2 + By**2 + Cz**2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz - 1 = 0
    Also known as
               ax**2 + by**2 + cz**2 +  dxy +  exz +  fyz +  gx +  hy +  iz + j = 0
    The equation of the ellipsoid in homogeneous coordinates is given by
                   [p0 p3 p4 p6]   [x]
                   [p3 p1 p5 p7]   [y]
    [x, y, z, 1] * [p4 p5 p2 p8] * [z] = 0
                   [p6 p7 p8 p9]   [1]
    Without homogeneous coordinates
                [p0 p3 p4]   [x]                 [p6]
    [x, y, z] * [p3 p1 p5] * [y]  + 2 *[x y z] * [p7] + p9 = 0
                [p4 p5 p2]   [z]                 [p8]
    Copyright:
        2011 Levente Hunyadi
        2024 Paul Leroy for the translation in Python

    :param p: the 10 parameters describing the ellipsoid algebraically
    :return center: ellipsoid center coordinates [cx, cy, cz]
    :return ax: ellipsoid semi-axes (radii) [a, b, c]
    :return quaternions: ellipsoid rotation in quaternion representation
    :return rotation_matrix: ellipsoid rotation (radii directions as rows of the 3x3 matrix)
    """

    # eliminate times two from rotation and translation terms
    p[3:9] = 0.5 * p[3:9]

    # find the algebraic form of the ellipsoid (quadratic form matrix)
    q = np.array([[p[0], p[3], p[4], p[6]],
                 [p[3], p[1], p[5], p[7]],
                 [p[4], p[5], p[2], p[8]],
                 [p[6], p[7], p[8], p[9]]])

    # find the center of the ellipsoid
    center = np.linalg.lstsq(q[0:3, 0:3], -p[6:9], rcond=None)[0]

    # form the corresponding translation matrix
    t = np.eye(4)
    t[3, 0: 3] = center

    # translate to the center
    s = t @ q @ t.T

    # check for positive definiteness
    # will raise LinAlgError if the decomposition fails, for example, if a is not positive-definite
    _ = np.linalg.cholesky(-s[3, 3] * s[0: 3, 0: 3])

    # solve the eigen problem
    eigenvalues, eigenvectors = np.linalg.eig(s[0: 3, 0: 3])
    radii = np.sqrt(-s[3, 3] / eigenvalues)

    # convert rotation matrix to quaternions
    quaternions = scipy.spatial.transform.Rotation.from_matrix(eigenvectors).as_quat()
    rotation_matrix = eigenvectors.T

    return center, radii, quaternions, rotation_matrix


def direct_fit(xyz, method='evd'):
    """
    Direct least squares fitting of ellipsoids under the constraint 4J - I**2 > 0.
    The constraint confines the class of ellipsoids to fit to those whose smallest radius
    is at least half of the largest radius.
    Reference:
        Qingde Li and John G. Griffiths, "Least Squares Ellipsoid Specific Fitting",
        Proceedings of the Geometric Modeling and Processing, 2004.
    Copyright:
        2011 Levente Hunyadi
        2024 Paul Leroy for the translation in Python

    :param xyz: coordinates of the 3D points
    :param method: specify the method to use: 'evd'
    :return: 10-parameter vector of the algebraic ellipsoid fit
    """
    x, y, z = np.split(xyz, 3, 1)

    # build design matrix
    d = np.c_[x**2, y**2, z**2, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z, np.ones(x.shape)]

    # build scatter matrix
    s = d.T @ d

    # build 10x10 constraint matrix
    k = 4  # to ensure that the parameter vector always defines an ellipse
    c1 = np.array([[0, k, k], [k, 0, k], [k, k, 0]]) / 2 - 1
    c2 = -k * np.eye(3)
    c3 = np.zeros((4, 4))

    if method == 'evd':
        c = scipy.linalg.block_diag(c1, c2, c3)

        # solve eigen system
        eigenvalues, eigenvectors = scipy.linalg.eig(s, c)

        # extract eigenvector corresponding to the unique positive eigenvalue
        flt = np.where((eigenvalues > 0) & (~np.isinf(eigenvalues)))[0]
        if len(flt) == 1:  # regular case
            v = eigenvectors[:, flt[0]]
        elif len(flt) == 0:  # degenerate case
            # single positive eigenvalue becomes near-zero negative eigenvalue due to round-off error
            ix = np.argmin(np.abs(eigenvalues))
            v = eigenvectors[:, ix]
        else:  # degenerate case
            # several positive eigenvalues appear
            ix = np.argmin(np.abs(eigenvalues))
            v = eigenvectors[:, ix]

    p = np.zeros(v.shape)
    p[0: 3] = v[0: 3]
    p[3: 6] = 2 * v[5: 2: -1]  # exchange order of y*z, x*z, x*y to x*y, x*z, y*z
    p[6: 9] = 2 * v[6: 9]
    p[9] = v[9]

    return p


def fit_ellipsoid_to_grain(xyz, method='direct'):
    # Shift point cloud to have only positive coordinates (problem with
    # quadfit if the point cloud is far from the coordinates of the origin (0,0,0))
    max_x, max_y, max_z = np.amax(xyz, axis=0)
    min_x, min_y, min_z = np.amin(xyz, axis=0)
    # find the scaling factor
    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z
    scale = 1 / max(dx, dy, dz)
    if method == 'direct':
        # Direct least squares fitting of ellipsoids under the constraint 4J - I**2 > 0.
        # The constraint confines the class of ellipsoids to fit to those whose smallest radius is at least half of the
        # largest radius.
        # Ellipsoid fit
        ellipsoid_parameters = direct_fit(scale * (xyz - np.mean(xyz, axis=0)))
        # Get the explicit parameters
        [center, radii, quaternions, rotation_matrix] = implicit_to_explicit(ellipsoid_parameters)
    else:
        raise ValueError('Unknown method')

    # Rescale the explicit parameters (the quaternions and R are unchanged by the scaling)
    center = center / scale + np.mean(xyz, axis=0)
    radii = radii / scale

    # Recompute the implicit form of the ellipsoid
    ellipsoid_parameters = explicit_to_implicit(center, radii, rotation_matrix)

    return center, radii, quaternions, rotation_matrix, ellipsoid_parameters


def ellipsoid(xc, yc, zc, xr, yr, zr):
    """
    The returned ellipsoid has center coordinates at (xc,yc,zc), semiaxis lengths (xr,yr,zr), and consists of 20-by-20
     faces.
    :param xc: center coordinate
    :param yc:
    :param zc:
    :param xr: semi-axis length
    :param yr:
    :param zr:
    :return:
    """

    number_of_faces = 20

    theta = np.linspace(0, np.pi, number_of_faces + 1)
    phi = np.linspace(0, 2 * np.pi, number_of_faces + 1)
    thetav, phiv = np.meshgrid(theta, phi)

    xx = xc + xr * np.sin(thetav) * np.cos(phiv)
    yy = yc + yr * np.sin(thetav) * np.sin(phiv)
    zz = zc + zr * np.cos(thetav)

    return xx, yy, zz


def plot_ellipsoid_fireframe(xc_yc_zc_xr_yr_zr=[0, 0, 0, 1, 1, 1]):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make data
    xx, yy, zz = ellipsoid(*xc_yc_zc_xr_yr_zr)

    # Plot a basic wireframe
    ax.plot_wireframe(xx, yy, zz)

    ax.axis('equal')

    plt.show()


def plot_ellipsoid_surface(xc_yc_zc_xr_yr_zr=[0, 0, 0, 1, 1, 1]):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data
    xx, yy, zz = ellipsoid(*xc_yc_zc_xr_yr_zr)

    # Plot the surface
    surf = ax.plot_surface(xx, yy, zz,
                           cmap=cm.coolwarm)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.axis('equal')

    plt.show()
