import sys
import time

from tqdm import trange
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import reconstruct as rec

sys.path.append('/Users/46h/Research/')
from accphys.tools import plotting as myplt
from accphys.tools import utils

savefig_kws = dict(dpi=300)


def get_weights(X, projection, screen_xedges, screen_yedges, normalize=False):
    n_parts = X.shape[0]
    n_bins_x = screen_xedges.size - 1
    n_bins_y = screen_yedges.size - 1
    xidx = np.digitize(X[:, 0], screen_xedges) - 1
    yidx = np.digitize(X[:, 2], screen_yedges) - 1
    on_screen = np.logical_and(np.logical_and(xidx >= 0, xidx < n_bins_x),
                               np.logical_and(yidx >= 0, yidx < n_bins_y))
    weights = np.zeros(n_parts)
    weights[on_screen] = projection[xidx[on_screen], yidx[on_screen]]
    if normalize:
        weights /= np.sum(weights)
    return weights


def compare_dist(coords_at_screen, measured_projections, screen_xedges, screen_yedges):
    cost = 0.
    for _X, projection in zip(coords_at_screen, measured_projections):
        Z, _, _ = np.histogram2d(_X[:, 0], _X[:, 2], bins=(screen_xedges, screen_yedges), density=True)
        cost += np.sum((Z - projection)**2)
    return cost / len(measured_projections)


# Setup
#------------------------------------------------------------------------------
print('Creating true distribution.')
# Create a rigid rotating distribution.
X = np.random.normal(size=(100000, 4))
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
n_bins = 40
axes = myplt.corner(X, figsize=(6, 6), bins=n_bins, cmap='viridis', pad=0.0)

# Store the limits for each dimension.
limits = [ax.get_xlim() for ax in axes[-1, :]]
labels = ["x", "x'", "y", "y'"]

# 4D histogram
Z_true, edges = np.histogramdd(X, n_bins, limits, density=True)
centers = []
for _edges in edges:
    centers.append(0.5 * (_edges[:-1] + _edges[1:]))

widths = np.array([abs(np.diff(centers[i])[0]) for i in range(4)])


# Simulate the measurements.
#------------------------------------------------------------------------------
print('Simulating measurements.')
K = 3 # number of angles in x dimension
L = 3 # number of angles in y dimension
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

screen_xedges = edges[0]
screen_yedges = edges[2]


# Reconstruction
#------------------------------------------------------------------------------
n_proj = len(projections)
n_parts = int(1e6)

# Create the initial distribution.
print('Creating seed distribution ({:.3e} particles.)'.format(n_parts))
scale = 10.0
lo = [scale * limit[0] for limit in limits]
hi = [scale * limit[1] for limit in limits]
X = np.random.uniform(lo, hi, size=(n_parts, 4))


max_iters = 10
atol = 10.0
rtol = 1e-2

old_cost = np.inf

for iteration in range(max_iters):

    # Re-discretize the measured profiles at the grid size setting and
    # normalize them. (Skip this step for now; maintain the same grid size on all
    # iterations.)

    # Track the distribution to the screen.
    # print('Tracking distribution to screen.')
    coords_at_screen = []
    for proj_index in range(n_proj):
        M = tmats[proj_index]
        X_screen = np.apply_along_axis(lambda row: np.matmul(M, row), 1, X)
        coords_at_screen.append(X_screen)

    # Get particle weights
    # print('Computing particle weights.')
    weights = np.zeros((X.shape[0], n_proj))
    for proj_index in range(n_proj):
        projection = projections[proj_index]
        X_screen = coords_at_screen[proj_index]
        weights[:, proj_index] = get_weights(X_screen, projection, screen_xedges, screen_yedges,
                                             normalize=False)

    # Compare the x-y projection of the distribution to the measurements.
    # print('Computing cost function.')
    cost = compare_dist(coords_at_screen, projections, screen_xedges, screen_yedges)
    dcost = cost - old_cost
    dcost_rel = np.inf if iteration == 0 else dcost / old_cost
    print('{} {:.3e} {:.3e} {:.3e} {:.3e}'.format(iteration, cost, dcost, dcost_rel, X.shape[0]))
    # if rtol > abs(dcost_rel):
    #     print('rtol satisfied')
    #     break
    # elif atol > abs(dcost):
    #     print('atol satisfied')
    #     break
    old_cost = cost


    # For each particle, if all weights are nonzero, compute the total weight by summing
    # weight of each projection; otherwise, set the total weight to zero.
    zero_row_idx, zero_col_idx = np.where(weights == 0.)
    weights[zero_row_idx, :] = 0.0
    weights = np.sum(weights, axis=1)
    weights /= np.sum(weights)

    # Get rid of the particles with zero total weight.
    keep_idx = weights > 0.
    weights = weights[keep_idx]
    X = X[keep_idx, :]
    print('There are {:.3e} particles with nonzero weights.'.format(X.shape[0]))

    # print('Generating new particles.')
    # Generate new particles in the neighborhood of the remaining particles.
    # The number of new particles is proportional to the weight.
    lo = -0.5 * widths
    hi = +0.5 * widths



    nmax = 1e5
    new_coords = []
    for i, weight in enumerate(weights):
        n = int(nmax * weight)
        dX = np.random.uniform(lo, hi, size=(n, 4))
        new_coords.append(X[i, :] + dX)

    X = np.vstack([X, *new_coords])
    print('The new bunch has {} particles.'.format(X.shape[0]))






Z, _ = np.histogramdd(X, n_bins, limits, density=True)

plot_kws = dict(cmap='viridis', shading='auto')
indices = [(0, 1), (2, 3), (0, 2), (0, 3), (2, 1), (1, 3)]
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(3.5, 9.25),
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
