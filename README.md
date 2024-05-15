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

## How it works

First you have to import the ```g3point``` module.  
**Note:** it's up to you to configure correctly the python path for your 
system to be able to find it.

```
import g3point
```

To instantiate a G3Point object, you will need a point cloud, in las or ply and an ini file (see the example in the 
```data``` 
section).

```
g3point_data = g3point.G3Point(cloud, ini)
```

### Initial segmentation

The initial segmentation is done once and for all, it is not modified by the clustering or by the cleaning. IT sets 
the following attributes:

- ```initial_labels```
- ```initial_sink_indexes```
- ```initial_stacks```

At the end of the initial segmentation, the variables ```labels```, ```sink_indexes``` and ```stacks``` are initialized to their 
```initial_``` counterparts.

```
g3point_data.initial_segmentation()
```

### Cluster

The clustering is based on the initial segmentation.

```
g3point_data.cluster()
```

### Clean

The cleaning is based on the current state of the labels

```
g3point_data.clean()
```

### Save the results

```
out, out_sinks = g3point_data.save()
```
