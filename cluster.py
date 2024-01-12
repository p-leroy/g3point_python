import numpy as np
import numpy.matlib
import scipy
from sklearn.cluster import DBSCAN


def angle_rot_2_vec_mat(a, b):

    c = np.zeros(max(a.shape, b.shape))
    c[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    c[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    c[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    d = np.sum(a * b, 1)

    angle = np.arctan2(np.linalg.norm(c, 2), d) * 180 / np.pi  # norm None => Frobenius norm, 2 => 2-norm (largest sing. value)

    return angle


def cluster_labels(xyz, params, neighbors_indexes, labels, nlabels, stacks, ndon, sink_indexes, surface, normals):

    print(f'[cluster_labels]')
    nlabels_start = nlabels

    # Compute the distances between sinks associated to each label
    D1 = scipy.spatial.distance.cdist(xyz[sink_indexes, :], xyz[sink_indexes, :])
    # Radius of each label (assuming the surface corresponds to a disk)
    A = np.zeros((1, nlabels))
    for k, stack in enumerate(stacks):
        A[0, k] = np.sum(surface[stack])

    radius = np.sqrt(A / np.pi)

    # Inter-distance by summing radius
    D2 = np.zeros((nlabels,nlabels))
    D2 = radius + radius.T
    ind = np.where(params.rad_factor * D2 > D1)
    Dist = np.zeros((nlabels, nlabels))
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
    for k in range(len(indborder)):
        i = indborder[k]  # i = index of the point
        j = neighbors_indexes[i, :]  # index of the neighbourhood of i
        # Take the normals vector for i and j (repmat on the normal vector for i to have the same size as for j)
        P1 = numpy.matlib.repmat(normals[i, :], params.knn, 1)
        P2 = normals[j, :]
        # Compute the angle between the normal of i and the normals of j
        # Add this angle to the angle matrix between each label
        A[labels[i], labels[j]] = A[labels[i], labels[j]] + angle_rot_2_vec_mat(P1, P2)
        # Number of occurences
        N[labels[i], labels[j]] = N[labels[i], labels[j]] + 1

    # Take the mean value
    # Aangle = A / N  # Aangle[np.where(N == 0)] = 0  may be needed to avoir NaNs
    Aangle = np.zeros(A.shape)
    N_not_null = np.where(N != 0)
    Aangle[N_not_null] = A[N_not_null] / N[N_not_null]

    # ---- Merge grains
    # Matrix of labels to be merged
    Mmerge = np.zeros((nlabels,nlabels))
    Mmerge[np.where((Dist < 1) | (Nneigh < 1) | (Aangle > params.max_angle1))] = 1e9
    np.fill_diagonal(Mmerge, 0)
    clustering = DBSCAN(eps=1, min_samples=1, metric='precomputed').fit(Mmerge)
    newlabels = np.zeros(labels.shape, dtype=int)
    nb_clusters = len(np.unique(clustering.labels_))
    new_stacks = [[] for k in range(nb_clusters)]
    for i in np.unique(clustering.labels_):
        ind = np.where(clustering.labels_ == i)[0]
        for j in ind:
            newlabels[stacks[j]] = i
            new_stacks[i] = new_stacks[i] + stacks[j]

    labels = newlabels
    nlabels = len(np.unique(labels))
    stacks = new_stacks
    nstack = [len(stack) for stack in stacks]

    sink_indexes = np.zeros(nlabels, dtype=int)
    for k, stack in enumerate(stacks):
        sink_index = np.argmax(xyz[stack, 2])
        sink_indexes[k] = stack[sink_index]

    print(f'[cluster_labels] check normals at the borders: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')

    return labels, nlabels, stacks, sink_indexes


def clean_labels(xyz, params, neighbors_indexes, labels, nlabels, stacks, ndon, sink_indexes, surface, normals):

    print('[clean_labels]')
    nlabels_start = nlabels

    # Determine if the normals at the border of labels are similars
    # Find the indexborder nodes (No bonor and many other labels in the Neighbourhood)
    temp = params.knn - np.sum(labels[neighbors_indexes] == np.tile(labels.reshape(-1, 1), params.knn), axis=1)
    indborder = np.where((temp >= params.knn / 4) & (ndon == 0))[0]
    # Compute the angle of the normal vector between the neighbours of each
    # grain/label
    A = np.zeros((nlabels, nlabels))
    N = np.zeros((nlabels, nlabels))
    for k in range(len(indborder)):
        i = indborder[k]  # i = index of the point
        j = neighbors_indexes[i, :]  # index of the neighbourhood of i
        # Take the normals vector for i and j (repmat on the normal vector for i to have the same size as for j)
        P1 = numpy.matlib.repmat(normals[i, :], params.knn, 1)
        P2 = normals[j, :]
        # Compute the angle between the normal of i and the normals of j
        # Add this angle to the angle matrix between each label
        A[labels[i], labels[j]] = A[labels[i], labels[j]] + angle_rot_2_vec_mat(P1, P2)
        # Number of occurences
        N[labels[i], labels[j]] = N[labels[i], labels[j]] + 1

    # Take the mean value
    Aangle = np.zeros(A.shape)
    N_not_null = np.where(N != 0)
    Aangle[N_not_null] = A[N_not_null] / N[N_not_null]

    # ---- Merge grains
    # Matrix of labels to be merged
    Mmerge = np.zeros((nlabels, nlabels))
    Mmerge[np.where((Aangle > params.max_angle2) | (Aangle == 0))] = 1e9
    np.fill_diagonal(Mmerge, 0)
    clustering = DBSCAN(eps=1, min_samples=1, metric='precomputed').fit(Mmerge)
    newlabels = np.zeros(labels.shape, dtype=int)
    nb_clusters = len(np.unique(clustering.labels_))
    new_stacks = [[] for k in range(nb_clusters)]
    for i in np.unique(clustering.labels_):
        ind = np.where(clustering.labels_ == i)[0]
        for j in ind:
            newlabels[stacks[j]] = i
            new_stacks[i] = new_stacks[i] + stacks[j]

    labels = newlabels
    nlabels = len(np.unique(labels))
    stacks = new_stacks
    nstack = np.array([len(stack) for stack in stacks])

    sink_indexes = np.zeros(nlabels, dtype=int)
    for k, stack in enumerate(stacks):
        sink_index = np.argmax(xyz[stack, 2])
        sink_indexes[k] = stack[sink_index]

    print(f'[clean_labels] check normals at the borders: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')
    nlabels_start = nlabels

    # remove small labels
    clusters_to_keep = np.where(nstack >= params.n_min)[0]
    newlabels = np.zeros(labels.shape, dtype=int)
    newisink = np.zeros(clusters_to_keep.shape, dtype=int)
    new_stacks = []
    for k, index in enumerate(clusters_to_keep):
        new_stacks.append(stacks[index])
        newisink[k] = sink_indexes[index]
        newlabels[stacks[index]] = k
    isink = newisink
    stacks = new_stacks
    labels = newlabels
    nlabels = len(np.unique(labels))
    print(f'[clean_labels] remove small labels: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')
    nlabels_start = nlabels

    # remove flattish labels (probably not grains)
    r = np.zeros((nlabels, 3))
    for k in range(nlabels):
        centroid = np.mean(xyz[stacks[k], :], axis=0)
        xyz_c = xyz[stacks[k], :] - centroid  # centered coordinates
        U, S, Vh = np.linalg.svd(xyz_c, full_matrices=False)  # singular value decomposition
        r[k, :] = S
    # filtering: (l2 / l0 > min_flatness) or (l1 / l0 > 2 * min_flatness)
    clusters_to_keep = np.where((r[:, 2] / r[:,  0] > params.min_flatness) | (r[:, 1]  /  r[:, 0] > 2. * params.min_flatness))[0]
    newlabels = np.zeros(labels.shape, dtype=int)
    newisink = np.zeros(clusters_to_keep.shape, dtype=int)
    new_stacks = []
    for k, index in enumerate(clusters_to_keep):
        new_stacks.append(stacks[index])
        newisink[k] = isink[index]
        newlabels[stacks[index]] = k
    isink = newisink
    stacks = new_stacks
    labels = newlabels
    nlabels = len(np.unique(labels))
    print(f'[clean_labels] remove flattish labels: {nlabels}/{nlabels_start} kept ({nlabels_start - nlabels} removed)')
    nlabels_start = nlabels

    labels[labels == 0] = -1

    return labels, nlabels, stacks, isink
