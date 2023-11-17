import numpy as np


def addtoStack(index, n_donors, donors, stack):
    # This recursive function adds to the stack the donors of an outlet, and then the donors of theses
    # donors and etc until an entire catchment is added to the stack

    stack.append(index)  # add nodes to the stack

    for k in range(n_donors[index]):  # find donors and add them (if any) to the stack
        addtoStack(donors[index, k], n_donors, donors, stack)


def segment_labels(xyz_detrended, knn, neighbors_indexes):
    print('[segment_labels]')

    # for each point, compute the slopes between the point and each one of its neighbors
    x, y, z = np.split(xyz_detrended, 3, axis=1)
    n_points = len(xyz_detrended)
    dx = x - np.squeeze(x[neighbors_indexes])  # squeeze removes axis of length 1
    dy = y - np.squeeze(y[neighbors_indexes])
    dz = z - np.squeeze(z[neighbors_indexes])
    slopes = dz / (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5  # compute the slopes between a point and its neighbors

    # for each point, find in the neighborhood the point with the minimum slope (the receiver)
    index_of_min_slope = np.argmin(slopes, axis=1)  # get the index of the point with the minimum slope
    slope = np.amin(slopes, axis=1)  # get the value of the minimum slope
    receivers = neighbors_indexes[np.arange(n_points), index_of_min_slope]

    # if the minimum slope is positive, we have a sink node (local maxima)
    sink_indexes = np.where(slope > 0)[0]
    receivers[sink_indexes] = sink_indexes

    # identify the donors for each receiver
    ndon = np.zeros(n_points, dtype=int)  # number of donors
    donor = np.zeros((n_points, knn), dtype=int)  # donor list
    for k, receiver in enumerate(receivers):
        if receiver != k:
            ndon[receiver] = ndon[receiver] + 1
            donor[receiver, ndon[receiver] - 1] = k

    # build the stacks
    labels = np.zeros(n_points, dtype=int)
    labelsk = np.zeros(n_points, dtype=int)
    labelsnpoint = np.zeros(n_points)
    stacks = []

    for k, ij in enumerate(sink_indexes):
        stack = []
        addtoStack(ij, ndon, donor, stack)  # recursive function
        stacks.append(stack)
        labels[stack] = k
        labelsnpoint[stack] = len(stack)

    nlabels = len(sink_indexes)

    return labels, nlabels, labelsnpoint, stacks, ndon, sink_indexes