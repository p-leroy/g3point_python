## Warning

### ellipsoid.implicit_to_explicit

When computing

center, radii, quaternions, rotation_matrix, ellipsoid_parameters = (
    ellipsoid.fit_ellipsoid_to_grain(xyz[stacks[0]]))

and then trying to launch

ellipsoid.implicit_to_explicit(ellipsoid_parameters)

twi times consecutively, the first time it works, the second time it does not work.