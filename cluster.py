import numpy as np
import numpy.matlib
import scipy
from sklearn.cluster import DBSCAN


def angle_rot_2_vec_mat(a, b, v2=False):
    c = np.zeros(max(a.shape, b.shape))
    c[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    c[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    c[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    d = np.sum(a * b, 1)

    if v2:
        angle = np.arctan2(np.linalg.norm(c, ord=None, axis=1),
                           d) * 180 / np.pi  # norm None => Frobenius norm, 2 => 2-norm (largest sing. value)
    else:  # this is the original computation from the Matlab code
        angle = np.arctan2(np.linalg.norm(c, 2),
                           d) * 180 / np.pi  # norm None => Frobenius norm, 2 => 2-norm (largest sing. value)

    return angle


def compute_mean_angle(params, labels, neighbors_indexes, ndon, normals, v2=False):

    nlabels = len(np.unique(labels))

    # Find the indexborder nodes (no donor and many other labels in the neighbourhood)
    temp = params.knn - np.sum(labels[neighbors_indexes] == np.tile(labels.reshape(-1, 1), params.knn), axis=1)
    indborder = np.where((temp >= params.knn / 4) & (ndon == 0))[0]

    # Compute the angle of the normal vector between the neighbours of each grain / label
    A = np.zeros((nlabels, nlabels))
    N = np.zeros((nlabels, nlabels))
    for i in indborder:  # i = index of the point
        j = neighbors_indexes[i, :]  # indexes of the neighbourhood of i
        # Take the normals vector for i and j (duplicate the normal vector of i to have the same size as for j)
        P1 = numpy.tile(normals[i, :], (params.knn, 1))
        P2 = normals[j, :]
        # Compute the angle between the normal of i and the normals of j
        if v2:
            for k, n in enumerate(j):
                A[labels[i], labels[n]] = A[labels[i], labels[n]] + angle_rot_2_vec_mat(P1, P2, v2=v2)[k]
                N[labels[i], labels[n]] = N[labels[i], labels[n]] + 1
        else:
            # Add this angle to the angle matrix between each label
            A[labels[i], labels[j]] = A[labels[i], labels[j]] + angle_rot_2_vec_mat(P1, P2, v2=v2)
            # Number of occurrences
            N[labels[i], labels[j]] = N[labels[i], labels[j]] + 1

    # Take the mean value
    # Aangle = np.zeros(A.shape)
    Aangle = np.empty(A.shape)
    Aangle.fill(numpy.nan)
    N_not_null = np.where(N != 0)
    Aangle[N_not_null] = A[N_not_null] / N[N_not_null]

    return Aangle


def merge_labels(labels, stacks, condition, symmetric_condition=False):

    # should we make the condition symmetric?
    if symmetric_condition:
        condition = condition & condition.T  # force the symmetry of the matrix

    nlabels = len(np.unique(labels))

    Mmerge = np.zeros((nlabels, nlabels))  # Matrix of labels to be merged
    Mmerge[np.where(condition)] = 1e9
    np.fill_diagonal(Mmerge, 0)

    clustering = DBSCAN(eps=1, min_samples=1, metric='precomputed').fit(Mmerge)
    new_labels = np.zeros(labels.shape, dtype=int)
    nb_clusters = len(np.unique(clustering.labels_))
    new_stacks = [[] for k in range(nb_clusters)]
    for i in np.unique(clustering.labels_):
        ind = np.where(clustering.labels_ == i)[0]
        for j in ind:
            new_labels[stacks[j]] = i
            new_stacks[i] = new_stacks[i] + stacks[j]

    return new_labels, new_stacks


def merge_labels_v2(labels, stacks, condition, condition_flag=None):

    if condition_flag == 'lower':
        # keep only the lower triangle of the condition matrix
        tmp = np.ones(condition.shape, dtype=bool)
        condition = np.tril(condition) | np.triu(tmp)
    elif condition_flag == 'upper':
        # keep only the upper triangle of the condition matrix
        tmp = np.ones(condition.shape, dtype=bool)
        condition = np.triu(condition) | np.tril(tmp)
    elif condition_flag == 'symmetrical':
        symmetrical_condition = np.ones(condition.shape, dtype=bool)
        indexes = np.where(condition == condition.T)
        symmetrical_condition[indexes] = condition[indexes]
        condition = symmetrical_condition

    n_labels = len(stacks)
    new_labels = np.ones(labels.shape, dtype=int) * (-1)
    count_new_labels = 0
    new_stacks = []

    for label in range(n_labels):

        if new_labels[label] == -1:
            new_labels[label] = count_new_labels
            new_stacks.append(stacks[label])
            count_new_labels = count_new_labels + 1

        for otherLabel in range(n_labels):

            if otherLabel == label:  # do not try to merge a label with itself
                continue

            if not condition[label, otherLabel]:
                if new_labels[otherLabel] != -1:  # the other label has already been merged
                    if new_labels[label] > new_labels[otherLabel]:  # merge label in OtherLabel
                        # add the label stack to the otherLabel stack
                        new_stacks[new_labels[otherLabel]] += new_stacks[new_labels[label]]
                        # empty the label stack
                        new_stacks[new_labels[label]] = []
                        # update the label
                        new_labels[label] = new_labels[otherLabel]
                    if new_labels[label] < new_labels[otherLabel]:  # merge otherLabel in label
                        # add the otherLabel stack to the label stack
                        new_stacks[new_labels[label]] += new_stacks[new_labels[otherLabel]]
                        # empty the otherLabel stack
                        new_stacks[new_labels[otherLabel]] = []
                        # update the label
                        new_labels[otherLabel] = new_labels[label]
                else:  # merge otherLabel and label
                    new_stacks[new_labels[label]] += stacks[otherLabel]
                    new_labels[otherLabel] = new_labels[label]

    # remove empty stacks
    if new_stacks.count([]):
        print(f"there are {new_stacks.count([])} empty stacks, remove them")
    else:
       print("no empty stack")

    new_stacks = [stack for stack in new_stacks if stack != []]

    # redefine labels
    new_labels = np.ones(labels.shape, dtype = int) * (-1)
    for k, stack in enumerate(new_stacks):
        for index in stack:
            new_labels[index] = k

    return new_labels, new_stacks


def cluster_labels(xyz, params, neighbors_indexes, labels, stacks, ndon, sink_indexes, surface, normals,
                   v2=False, my_merge=True, condition_flag=None):
    print(f'[cluster_labels]')
    nlabels = len(np.unique(labels))
    nlabels_start = nlabels

    # Compute the distances between sinks associated to each label
    D1 = scipy.spatial.distance.cdist(xyz[sink_indexes, :], xyz[sink_indexes, :])

    # Radius of each label (assuming the surface corresponds to a disk)
    A = np.zeros((1, nlabels))
    for k, stack in enumerate(stacks):
        A[0, k] = np.sum(surface[stack])
    radius = np.sqrt(A / np.pi)
    D2 = radius + radius.T  # Inter-distance by summing radius

    # If the radius of the sink is above the distance to the other sink (by a factor of rad_factor), set Dist to 1
    ind = np.where(params.rad_factor * D2 > D1)
    Dist = np.zeros((nlabels, nlabels))
    Dist[ind] = 1
    Dist = Dist - np.eye(len(Dist))

    # If labels are neighbours, set Nneigh to 1
    Nneigh = np.zeros((nlabels, nlabels))
    for k, stack in enumerate(stacks):
        ind = np.unique(labels[neighbors_indexes[stack, :]])
        Nneigh[k, ind] = 1

    Aangle = compute_mean_angle(params, labels, neighbors_indexes, ndon, normals, v2=v2)

    # Merge labels if:
    # => sinks are close to each other (Dist == 1)
    # => sinks are neighbours (Nneigh == 1)
    # => normals are similar
    if v2:
        labels, stacks = merge_labels_v2(
            labels, stacks,
            (Dist < 1) | (Nneigh < 1) | (Aangle > params.max_angle1) | (np.isnan(Aangle)),
            condition_flag=None)
    else:
        if my_merge:
            labels, stacks = merge_labels_v2(
                labels, stacks,
                (Dist < 1) | (Nneigh < 1) | (Aangle > params.max_angle1) | (np.isnan(Aangle)),
                condition_flag=condition_flag)
        else:
            labels, stacks = merge_labels(
                labels, stacks,
                (Dist < 1) | (Nneigh < 1) | (Aangle > params.max_angle1) | (np.isnan(Aangle)))

    nlabels = len(np.unique(labels))

    sink_indexes = get_sink_indexes(stacks, xyz)

    print(
        f'[cluster_labels] check normals at the borders: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')

    return labels, nlabels, stacks, sink_indexes


def keep_labels(labels, stacks, condition, sink_indexes):

    clusters_to_keep = np.where(condition)[0]
    new_labels = np.zeros(labels.shape, dtype=int)
    new_sink_indexes = np.zeros(clusters_to_keep.shape, dtype=int)
    new_stacks = []
    for k, index in enumerate(clusters_to_keep):
        new_stacks.append(stacks[index])
        new_sink_indexes[k] = sink_indexes[index]
        new_labels[stacks[index]] = k

    return new_labels, new_stacks, new_sink_indexes


def get_sink_indexes(stacks, xyz):

    nlabels = len(stacks)
    sink_indexes = np.zeros(nlabels, dtype=int)
    for k, stack in enumerate(stacks):  # compute sink_indexes
        sink_index = np.argmax(xyz[stack, 2])
        sink_indexes[k] = stack[sink_index]

    return sink_indexes


def check_stacks(stacks, number_of_points):

    check = True
    # Initialize the set of indexes with the first stack
    stack = stacks[0]
    myset = {*stack}
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
        check = False
        raise ValueError('stacks are not coherent: the length of the set shall be equal to the number of points')
    if min != 0:  # min value in the set
        check = False
        raise ValueError('stacks are not coherent: min shall be 0')
    if max != (number_of_points - 1):  # max value in the set
        check = False
        raise ValueError('stacks are not coherent: max shall be equal to (number_of_points - 1)')

    return check


def clean_labels(xyz, params, neighbors_indexes, labels, stacks, ndon, normals):

    print('[clean_labels]')
    nlabels_start = len(np.unique(labels))

    Aangle = compute_mean_angle(params, labels, neighbors_indexes, ndon, normals)

    # Merge grains
    condition = (Aangle > params.max_angle2) | (Aangle == 0)
    labels, stacks = merge_labels(labels, stacks, condition)
    nlabels = len(np.unique(labels))
    nstack = np.array([len(stack) for stack in stacks])

    sink_indexes = get_sink_indexes(stacks, xyz)

    print(
        f'[clean_labels] check normals at the borders: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')
    nlabels_start = nlabels

    # remove small labels
    condition = (nstack >= params.n_min)
    labels, stacks, sink_indexes = keep_labels(labels, stacks, condition, sink_indexes)

    nlabels = len(np.unique(labels))
    print(f'[clean_labels] remove small labels: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')
    nlabels_start = nlabels

    # remove flattish labels (probably not grains)
    r = np.zeros((nlabels, 3))
    for k, stack in enumerate(stacks):
        centroid = np.mean(xyz[stack, :], axis=0)  # compute the centroid of the label
        xyz_c = xyz[stack, :] - centroid  # centered coordinates
        U, S, Vh = np.linalg.svd(xyz_c, full_matrices=False)  # singular value decomposition
        r[k, :] = S
    # filtering condition: (l2 / l0 > min_flatness) or (l1 / l0 > 2 * min_flatness)
    condition = (r[:, 2] / r[:, 0] > params.min_flatness) | (r[:, 1] / r[:, 0] > 2. * params.min_flatness)
    labels, stacks, sink_indexes = keep_labels(labels, stacks, condition, sink_indexes)

    nlabels = len(np.unique(labels))
    print(f'[clean_labels] remove flattish labels: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')

    labels[labels == 0] = -1

    return labels, nlabels, stacks, sink_indexes
