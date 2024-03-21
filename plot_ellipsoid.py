import matplotlib.pyplot as plt

from ellipsoid import ellipsoid

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xc_yc_zc_xr_yr_zr = [0, 0, 0, 1, 1, 1]

xx, yy, zz = ellipsoid(*xc_yc_zc_xr_yr_zr)

ax.plot_surface(xx, yy, zz)
