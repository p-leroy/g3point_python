import scipy

#%% Read matrix
indNeighbors = scipy.io.loadmat(r'C:\DATA\PhilippeSteer\G3Point\indNeighbors.mat')['indNeighbors']

#%% Compare
i, j = np.where((indNeighbors - neighbors_indexes - 1) != 0)
print(f'{len(i)} differences between Matlab and Python nearest neighbors indexes')

#%% Read distances matrix (D)
D = scipy.io.loadmat(r'C:\DATA\PhilippeSteer\G3Point\D.mat')['D']

#%% Compare => the points are at the same distance, this is just the order which changes
D[i, j]
neighbors_distances[i, j]

#%% Read stack
stack = scipy.io.loadmat(r'C:\DATA\PhilippeSteer\G3Point\stack.mat')['stack']

#%% Compare
stacks[0] - stack[0, 0] + 1
stacks[1] - stack[0, 1] + 1
stacks[10] - stack[0, 10] + 1

#%%
