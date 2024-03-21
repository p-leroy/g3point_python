# G3Point

Granulometry from 3D Point clouds
> Based on Steer, Guerit et al. (2022)

![ForGitHub](https://user-images.githubusercontent.com/17555304/159018713-7272a95e-6400-4490-83f5-868248cffbcb.gif)

## Introduction

**G3Point** is a tool which aims at automatically measuring the size, shape, and orientation of a large 
number of individual grains as detected from any type of 3D point clouds describing the topography of surfaces covered by sediments.
The tool has been developped initially in *Matlab* https://github.com/philippesteer/G3Point

This repository aims at converting the tool in *Python* at first and also to try to improve it it a longer term.

This algorithm relies on 3 main phases:
1. Grain **segmentation** using a waterhsed algorithm
2. Grain **merging and cleaning**
3. Grain **fitting by geometrical models** including ellipsoids and cuboids

## Warning

### ellipsoid.implicit_to_explicit

When computing

center, radii, quaternions, rotation_matrix, ellipsoid_parameters = (
    ellipsoid.fit_ellipsoid_to_grain(xyz[stacks[0]]))

and then trying to launch

ellipsoid.implicit_to_explicit(ellipsoid_parameters)

twi times consecutively, the first time it works, the second time it does not work.