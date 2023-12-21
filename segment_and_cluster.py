from time import perf_counter

import numpy as np
import scipy


def add_to_stack(index, n_donors, donors, stack):
    # This recursive function adds to the stack the donors of an outlet, and then the donors of these
    # donors, etc. Until an entire catchment is added to the stack

    stack.append(index)  # add nodes to the stack

    for k in range(n_donors[index]):  # find donors and add them (if any) to the stack
        add_to_stack(donors[index, k], n_donors, donors, stack)


def add_to_stack_bw(index, delta, Di, stack, local_maximum):
    # This recursive function adds to the stack the donors of an outlet, and then the donors of these
    # donors, etc. Until an entire catchment is added to the stack

    stack.append(index)  # add nodes to the stack

    for k in range(delta[index], delta[index + 1]):  # find donors and add them (if any) to the stack
        if Di[k] != local_maximum:  # avoid infinite loop
            add_to_stack_bw(Di[k], delta, Di, stack, local_maximum)


def philippe_steer_stack_building(receivers, local_maximum_indexes, knn):
    start = perf_counter()
    n_points = len(receivers)

    # identify the donors for each receiver
    ndon = np.zeros(n_points, dtype=int)  # number of donors
    donor = np.zeros((n_points, knn), dtype=int)  # donor list
    for k, receiver in enumerate(receivers):
        if receiver != k:
            ndon[receiver] = ndon[receiver] + 1
            donor[receiver, ndon[receiver] - 1] = k

    # build the stacks
    labels = np.zeros(n_points, dtype=int)
    labelsnpoint = np.zeros(n_points)
    stacks = []

    for k, ij in enumerate(local_maximum_indexes):
        stack = []
        add_to_stack(ij, ndon, donor, stack)  # recursive function
        stacks.append(stack)
        labels[stack] = k
        labelsnpoint[stack] = len(stack)

    end = perf_counter()
    print(f'Philippe Steer {end - start}')

    return stacks, labels, labelsnpoint, ndon


def braun_willett_stack_building(receivers, local_maximum_indexes):
    start = perf_counter()
    n_points = len(receivers)

    # get the number of donors per receiver and build the list of donors per receiver
    di = np.zeros(n_points, dtype=int)  # number of donors
    Dij = [[] for i in range(n_points)]  # lists of donors
    for k, receiver in enumerate(receivers):
        di[receiver] = di[receiver] + 1
        Dij[receiver].append(k)

    # build Di, the list of donors
    Di = np.zeros(n_points, dtype=int)  # list of donors
    idx = 0
    for list_ in Dij:  # build the list of donors
        for point in list_:
            Di[idx] = point
            idx = idx + 1

    # build delta, the index array
    delta = np.zeros(n_points + 1, dtype=int)  # index of the first donor
    delta[n_points] = n_points
    for i in range(n_points - 1, -1, -1):
        delta[i] = delta[i + 1] - di[i]
    stacks = []
    labels = np.zeros(n_points, dtype=int)
    points_per_label = np.zeros(n_points, dtype=int)

    # build the stacks
    for k, ij in enumerate(local_maximum_indexes):
        stack = []
        add_to_stack_bw(ij, delta, Di, stack, ij)  # recursive function
        stacks.append(stack)
        labels[stack] = k
        points_per_label[stack] = len(stack)

    end = perf_counter()
    print(f'Braun Willet {end - start}')

    return stacks, labels, points_per_label, di


def segment_labels(xyz, knn, neighbors_indexes, braun_willett=True):
    print('[segment_labels]')

    # for each point, compute the slopes between the point and each one of its neighbors
    x, y, z = np.split(xyz, 3, axis=1)
    n_points = len(xyz)
    dx = x - np.squeeze(x[neighbors_indexes])  # squeeze removes axis of length 1
    dy = y - np.squeeze(y[neighbors_indexes])
    dz = z - np.squeeze(z[neighbors_indexes])
    slopes = dz / (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5  # compute the slopes between a point and its neighbors

    # for each point, find in the neighborhood the point with the minimum slope (the receiver)
    index_of_min_slope = np.argmin(slopes, axis=1)  # get the index of the point with the minimum slope
    min_slope = np.amin(slopes, axis=1)  # get the value of the minimum slope
    receivers = neighbors_indexes[np.arange(n_points), index_of_min_slope]

    # if the minimum slope is positive, we have a local maximum
    local_maximum_indexes = np.where(min_slope > 0)[0]
    receivers[local_maximum_indexes] = local_maximum_indexes

    if braun_willett:
        stacks, labels, labelsnpoint, ndon = braun_willett_stack_building(receivers, local_maximum_indexes)
    else:
        stacks, labels, labelsnpoint, ndon = philippe_steer_stack_building(receivers, local_maximum_indexes, knn)

    nlabels = len(local_maximum_indexes)

    return labels, nlabels, labelsnpoint, stacks, ndon, local_maximum_indexes


def anglerot2vecmat(a,b):

    c = np.zeros(max(len(a),len(b)))
    c[1,:] = a[2, :] * b[3, :] - a[3, :] * b[2, :]
    c[2,:] = a[3, :] * b[1, :] - a[1, :] * b[3, :]
    c[3,:] = a[1, :] * b[2, :] - a[2, :] * b[1, :]

    d = np.sum(a * b, 1)

    angle = np.arctan2(np.linalg.norm(c), d)

    return angle


def cluster_labels(xyz, params, neighbors_indexes, labels, nlabels, stacks, ndon, sink_indexes, surface,normals):

    print('[cluster_labels]')

    # Compute the distances between sinks associated to each label
    D1 = scipy.spatial.distance.cdist(xyz[sink_indexes, :], xyz[sink_indexes, :])
    # Radius of each label (assuming the surface corresponds to a disk)
    A = np.zeros((1, nlabels))
    for k in range(nlabels):
        A[0, k] = np.sum(surface[stacks[k]])

    radius = np.sqrt(A / np.pi)

    # Inter-distance by summing radius
    D2 = np.zeros((nlabels,nlabels))
    D2 = radius + radius.T
    Dist = np.zeros((nlabels,nlabels))
    ind = np.where(params.rad_factor * D2 > D1)
    Dist[ind] = 1
    Dist = Dist - np.eye(len(Dist))

    # Determine if labels are neighbours
    Nneigh = np.zeros((nlabels,nlabels))
    for k in range(nlabels):
        ind = np.unique(labels[neighbors_indexes[stacks[k],:]])
        Nneigh[k, ind] = 1

    # Determine if the normals at the border of labels are similars
    # Find the indexborder nodes (No bonor and many other labels in the Neighbourhood)
    temp = params.knn - np.sum(labels[neighbors_indexes] == np.tile(labels.reshape(-1, 1), params.knn), axis=1)
    indborder = np.where((temp >= params.knn / 4) & (ndon==0))[0]
    # Compute the angle of the normal vector between the neighbours of each
    # grain/label
    A = np.zeros((nlabels,nlabels))
    N = np.zeros((nlabels,nlabels))
    # => CHECKED STEP BY STEP UNTIL THIS POINT
    for k in range(len(indborder)):
        # i = index of the point / j = index of the neighbourhood of i
        i = indborder[k]
        j = neighbors_indexes[i, :]
        # Take the normals vector for i and j (repmat on the normal vector for i to have the
        # same size as for j)
        P1 = np.matlib.repmat(normals[i, :], params.knn, 1)
        P2 = normals[j, :]
        # Compute the angle between the normal of i and the normals of j
        # Add this angle to the angle matrix between each label
        A[labels[i], labels[j]] = A[labels[i], labels[j]] + anglerot2vecmat(P1, P2)
        # Number of occurence
        N[labels[i], labels[j]] = N[labels[i], labels[j]] + 1

    # Take the mean value
    Aangle = A / N

    # ---- Merge grains
    # Matrix of labels to be merged
    Mmerge = np.zeros((nlabels,nlabels))
    Mmerge[np.where(Dist < 1 | Nneigh < 1 | Aangle > params.maxangle1)] = np.Inf

    [idx, _] = dbscan(Mmerge, 1, 1,'Distance', 'precomputed')
    newlabels = np.zeros(labels.shape)
    new_stacks = []
    for i in range(len(np.unique(idx))):
        ind = np.where(idx == i)
        for j in range(len(ind)):
            newlabels[stacks[ind[[j]]]] = i
            new_stacks.append(stack[ind[j]])

    labels = newlabels
    nlabels = max(labels)
    stacks = new_stacks
    nstack = [len(stack) for stack in new_stacks]

    for i in range(nlabels):
        temp = np.argmax(xyz(stacks[i], 3))
        sink_indexes[i] = stacks[i][temp]

    return labels, nlabels, stack, sink_indexes
