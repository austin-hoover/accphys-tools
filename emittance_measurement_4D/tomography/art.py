"""This script tests algebraic reconstruction (ART) in 4D phase space."""
import sys
import time

from tqdm import trange
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import reconstruct as rec

sys.path.append('/Users/46h/Research/')
from accphys.tools import plotting as myplt
from accphys.tools import utils


n_bins = 75


# Setup
#------------------------------------------------------------------------------
print('Creating distribution.')
# Create a rigid rotating distribution.
X = np.random.normal(size=(1000000, 4))
X = np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, X)
X[:, 3] = +X[:, 0]
X[:, 1] = -X[:, 2]

# Change the x-y phase difference.
R = np.zeros((4, 4))
R[:2, :2] = utils.rotation_matrix(np.pi / 4)
R[2:, 2:] = utils.rotation_matrix(0.0)
X = np.apply_along_axis(lambda row: np.matmul(R, row), 1, X)

# Add some noise.
X += np.random.normal(scale=0.4, size=X.shape)

# Plot the 2D projections.
axes = myplt.corner(X, figsize=(6, 6), bins=n_bins, cmap='viridis')

# Store the limits for each dimension.
limits = [ax.get_xlim() for ax in axes[-1, :]]
labels = ["x", "x'", "y", "y'"]

# 4D histogram
Z_true, edges = np.histogramdd(X, n_bins, limits, density=True)
centers = []
for _edges in edges:
    centers.append(0.5 * (_edges[:-1] + _edges[1:]))
bin_volume = rec.get_bin_volume(limits, n_bins)



# Simulate the measurements.
#------------------------------------------------------------------------------
print('Simulating measurements.')
K = L = 7 # number of angles in x/y dimension
n_proj = K * L
muxx = muyy = np.linspace(0., np.pi, K, endpoint=False)

xx_list = []
for mux in tqdm(muxx):
    Mx = utils.rotation_matrix(mux)
    xx_list.append(utils.apply(Mx, X[:, :2])[:, 0])
    
yy_list = []
for muy in tqdm(muyy):
    My = utils.rotation_matrix(muy)
    yy_list.append(utils.apply(My, X[:, 2:])[:, 0])
    
projections = []
for xx in tqdm(xx_list):
    for yy in yy_list:
        projection, _, _ = np.histogram2d(xx, yy, n_bins, (limits[0], limits[2]))
        projection = rec.process(projection, density=True, limits=[limits[0], limits[2]])
        projections.append(projection)

tmats = []
for mux in muxx:
    for muy in muyy:
        M = np.zeros((4, 4))
        M[:2, :2] = utils.rotation_matrix(mux)
        M[2:, 2:] = utils.rotation_matrix(muy)
        tmats.append(M)



screen_edges_x = edges[0]
screen_edges_y = edges[2]



# Reconstruction
#------------------------------------------------------------------------------
print('Forming arrays.')

# Treat each reconstruction bin center as a particle. We will call this collection of 
# particles the "bunch".
rec_grid_coords = rec.get_grid_coords(*centers)

# Keep this for later.
col_indices = np.arange(n_bins**4)


row_block_size = n_bins**2
rho = np.zeros(n_proj * row_block_size) # measured density on the screen.
rows, cols = [], [] # nonzero row and column indices of P

for proj_index in trange(n_proj):
    # Transport the bunch to the screen.
    M = tmats[proj_index]
    screen_grid_coords = np.apply_along_axis(lambda row: np.matmul(M, row), 1, rec_grid_coords)

    # For each particle, record the indices of the bin it landed in. So we want (k, l) such
    # that the particle landed in the bin with x = x[k] and y = y[l] on the screen. One of 
    # the indices will be -1 or n_bins if the particle landed outside the screen.
    xidx = np.digitize(screen_grid_coords[:, 0], screen_edges_x) - 1
    yidx = np.digitize(screen_grid_coords[:, 2], screen_edges_y) - 1
    on_screen = np.logical_and(np.logical_and(xidx >= 0, xidx < n_bins), 
                               np.logical_and(yidx >= 0, yidx < n_bins))
    
    # Get the indices for the flattened array.
    projection = projections[proj_index]
    screen_idx = np.ravel_multi_index((xidx, yidx), projection.shape, mode='clip')

    # P[i, j] = 1 if particle j landed in bin j on the screen, or 0 otherwise.
    i_offset = proj_index * row_block_size
    for j in tqdm(col_indices[on_screen]):
        i = screen_idx[j] + i_offset
        rows.append(i)
        cols.append(j)
    rho[i_offset: i_offset + row_block_size] = projection.flat


print('Creating sparse matrix P.')
t = time.time()
P = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_proj * n_bins**2, n_bins**4))
print('Done. t = {}'.format(time.time() - t))

print('Solving linear system.')
t = time.time()
psi, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = sparse.linalg.lsqr(P, rho, show=True, iter_lim=100)
print()
print('Done. t = {}'.format(time.time() - t))

print('Reshaping phase space density.')
Z = psi.reshape((n_bins, n_bins, n_bins, n_bins))
Z = np.clip(Z, 0.0, None)

print('Plotting.')
plot_kws = dict(cmap='viridis', shading='auto')
indices = [(0, 1), (2, 3), (0, 2), (0, 3), (2, 1), (1, 3)]
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(3.5, 9),
                         constrained_layout=True)
for row, (i, j) in enumerate(indices):
    _Z_true = rec.project(Z_true, [i, j])
    _Z = rec.project(Z, [i, j])
    axes[row, 0].pcolormesh(_Z.T, **plot_kws)
    axes[row, 1].pcolormesh(_Z_true.T, **plot_kws)
    axes[row, 0].annotate('{}-{}'.format(labels[i], labels[j]),
                          xy=(0.02, 0.92), xycoords='axes fraction', color='white')
for ax, title in zip(axes[0, :], ['Reconstructed', 'True', 'Error']):
    ax.set_title(title)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('_output/rec.png', dpi=300)


print('Saving reconstructed distribution.')
np.save('_output/Z.npz', Z)
np.save('_output/Z_true.npz', Z_true)
