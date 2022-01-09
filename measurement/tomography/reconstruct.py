"""Tomographic image reconstruction for beam diagnostics.

All angles should be kept in radians. We convert to degrees only when passing
the angles to skimage.
"""
import sys
import os
import time

import numpy as np
from scipy import sparse
from scipy import interpolate
from skimage.transform import iradon
from skimage.transform import iradon_sart
from skimage import filters
from tqdm import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import proplot as pplt


def apply(M, X):
    """Apply matrix M to each row of X."""
    return np.apply_along_axis(lambda row: np.matmul(M, row), 1, X)


def get_bin_centers(bin_edges):
    """Return bin centers from bin edges."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def get_bin_edges(bin_centers):
    """Return bin edges from bin centers."""
    w = 0.5 * np.diff(bin_centers)[0]
    return np.hstack([centers - w, [centers[-1] + w]])


def project(Z, indices):
    """Form projection of distribution.
    
    Parameters
    ----------
    Z : ndarray, n dimensions
        The distribution.
    indices : int, list, or tuple
        The distribution is projected onto these indices.
        
    Returns
    -------
    ndarray, n - len(indices) dimensions
        The projection of the distribution.
    """
    if type(indices) is int:
        indices = [indices]
    axis = tuple([k for k in range(Z.ndim) if k not in indices])
    return np.sum(Z, axis=axis)


def normalize(Z, bin_volume=1.0):
    """Normalize the distribution."""
    Zn = np.copy(Z)
    A = np.sum(Zn)
    if A == 0.0:
        return Zn
    return Zn / A / bin_volume


def get_bin_volume(limits, n_bins):
    """Return the bin volume.
    
    limits : list[tuple]
        (min, max) for each dimension.
    n_bins : int or list[int]
        The number of bins in each dimension.
    """
    if type(n_bins) is int:
        n_bins = len(limits) * [n_bins]
    return np.prod([(np.diff(lim)[0] / n) for lim, n in zip(limits, n_bins)])


def process(Z, keep_positive=False, density=False, limits=None):
    """Ensure distribution is normalized, positive, etc."""
    if keep_positive:
        Z = np.clip(Z, 0.0, None)
    if density:
        bin_volume = 1.0 
        if limits is not None:
            bin_volume = get_bin_volume(limits, Z.shape)
        Z = normalize(Z, bin_volume)
    return Z


def get_projection_angle(M):
    """Return projection angle from 2x2 transfer matrix.
    
    Let M propagate the phase space coordinates from point A to point B. The 
    projection onto the x axis is measured at B: p_B(x). In the phase space at
    point A, the projection is along the s axis, which is rotated by an angle 
    theta above the x axis. The angle is given by tan(theta) = M_{11} / M{12}.
    The value returned is in the range [0, 2pi].
    """
    theta = np.arctan(M[0, 1] / M[0, 0])
    if theta < 0.0:
        theta += np.pi
    return theta


def get_projection_scaling(M):
    """Return projection scaling from 2x2 transfer matrix.
    
    Let M propagate the phase space coordinates from point A to point B. The 
    projection onto the x axis is measured at B: p_B(x). In the phase space at
    point A, the projection is along the s axis, which is rotated by an angle 
    theta above the x axis. The projection axis is also changed. The s axis at 
    A and the x axis at B are related by x = r * s, where r = sqrt(M_{11}^2 + 
    M_{12}^2), meaning the projection is horizontally squeezed or stretched by 
    the transformation. Because of this, the projeciton needs to be vertically 
    scaled to maintain its area. Thus, the projections are related by p_A(s) =
    r * p_B(r * s). 
    """
    return np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    
    
def get_grid_coords(*xi, indexing='ij'):
    """Return array of shape (N, D), where N is the number of points on 
    the grid and D is the number of dimensions."""
    return np.vstack([X.ravel() for X in np.meshgrid(*xi, indexing='ij')]).T
    

def transform(Z, V, grid, new_grid=None):
    """Apply a linear transformation to a distribution.
    
    Note: this takes a while for high-dimensional distributions. Maybe
    there is a better way.
    
    Parameters
    ----------
    Z : ndarray, shape (len(x1), ..., len(xn))
         The distribution function in the original space.
    V : ndarray, shape (len(xi),)
        Matrix to transform the coordinates.
    grid : list[array_like]
        List of 1D arrays [x1, x2, ...] representing the bin centers in the 
        original space.
    new_grid : list[array_like] (optional)
        List of 1D arrays [x1, x2, ...] representing the bin centers in the 
        transformed space. If none are provided, we use the min/max coordinates
        of the transformed grid and keep the same number of bins.
        
    Returns
    -------
    Z : ndarray, shape (len(x1), ..., len(xn))
        The distribution function in the transformed space. Linear 
        interpolation is used to fill in the gaps.
    new_grid : list[array_like] 
        List of 1D arrays [x1, x2, ...] representing the bin centers in the 
        transformed space.
    """        
    # Transform the grid coordinates.
    coords_new = apply(V, get_grid_coords(*grid))
        
    # Define the interpolation coordinates.
    if new_grid is None:
        mins = np.min(coords_new, axis=0)
        maxs = np.max(coords_new, axis=0)
        new_grid = [np.linspace(mins[i], maxs[i], Z.shape[i]) 
                    for i in range(Z.ndim)]    
    coords_int = get_grid_coords(*new_grid)
    
    # Interpolate.
    Z = interpolate.griddata(coords_new, Z.ravel(), coords_int, 
                             method='linear', fill_value=0.0)
    Z = Z.reshape([len(xi) for xi in new_grid])
    return Z, new_grid


# 2D reconstruction
#------------------------------------------------------------------------------
def scale_projections(projections, tmats, centers_meas, centers_rec):
    """Given the projections at B and the linear transfer matrices from A to B, 
    return the projections at A.
    
    See `get_projection_scaling` and `get_projection_angle`. 
    
    Parameters
    ----------
    projections : list, shape (n_proj, N)
        Measured 1D projections of the distribution.
    tmats: list, shape (n_proj, 2, 2)
        Transfer matrices from A to B.
    centers_meas : list, shape (N,)
        Bin center coordinates at B.
    centers_rec : list, shape (M,)
        Bin center coordinates at A.
    
    Returns
    -------
    scaled_projections : list, shape (n_proj, M)
        Scaled projections in the phase space at A.
    proj_angles : list[float]
        Angles of the projections at A.
    """
    n_proj, n_bins = np.shape(projections)
    scaled_projections = np.zeros((n_proj, n_bins))
    proj_angles = np.zeros(n_proj)
    scale_factors = np.zeros(n_proj)
    for k, (M, projection) in enumerate(zip(tmats, projections)):
        r = get_projection_scaling(M)
        interp = interpolate.interp1d(centers_meas, projection, kind='linear', 
                                      bounds_error=False, fill_value=0.0)
        scaled_projections[k, :] = r * interp(r * centers_rec)
        scale_factors[k] = r
        proj_angles[k] = get_projection_angle(M)
    return scaled_projections, proj_angles


def rec2D(projections, tmats, centers_meas, centers_rec, 
          method='SART', proc_kws=None, **kws):
    """Reconstruct the x-x' or y-y' distribution from 1D projections.
    
    Parameters
    ----------
    projections : list, shape (n_proj, N)
        Measured 1D projections of the distribution.
    tmats: list, shape (n_proj, 2, 2)
        Transfer matrices from reconstruction point to measurement point.
    centers_meas : list, shape (N,)
        Bin center coordinates at the measurement point.
    centers_rec : list, shape (M,)
        Bin center coordinates at the reconstruction point.
    method : {'SART', 'FBP', 'MENT'}
        The reconstruction method to use.
    proc_kws : dict
        Key word arguments for `process`.
    **kws
        Key word arguments for reconstruction method.
        
    Returns
    -------
    Z : ndarray, shape (M, M)
        Reconstructed phase space distribution. The horizontal and vertical
        grid coordinates are the same: `xx_rec`. 
    """
    rfunc = None
    if method == 'SART':
        rfunc = sart
    elif method == 'FBP':
        rfunc = fbp
    elif method == 'MENT':
        rfunc = ment
    else:
        raise ValueError("Invalid reconstruction method.")
    if proc_kws is None:
        proc_kws = dict()
    projections, angles = scale_projections(projections, tmats, 
                                            centers_meas, centers_rec)
    angles = np.degrees(angles)
    Z = rfunc(projections, angles, **kws).T
    Z = process(Z, **proc_kws)
    return Z
    

def fbp(projections, angles, **kws):
    """Filtered back-projection (FBP)."""
    return iradon(projections.T, theta=-angles, **kws)
    
    
def sart(projections, angles, iterations=1, **kws):
    """Simultaneous algebraic reconstruction (SART)."""
    if 'iterations' in kws:
        iterations = kws.pop('iterations')
    Z = iradon_sart(projections.T, theta=-angles, **kws)
    for _ in range(iterations - 1):
        Z = iradon_sart(projections.T, theta=-angles, image=Z, **kws)
    return Z


def ment(projections, angles, proc_kws=None):
    """Maximum Entropy (MENT)."""
    raise NotImplementedError
    
    
# 4D reconstruction
#------------------------------------------------------------------------------
def hock4D(S, centers_meas, centers_rec, tmats_x, tmats_y, 
           method='SART', proc_kws=None, **kws):
    """4D reconstruction using method from Hock (2013).

    Parameters
    ----------
    S : ndarray, shape (n_bins, n_bins, n_proj, n_proj)
        Projection data. S[i, j, k, l] gives the intensity at (x[i], y[j]) on
        the screen for transfer matrix M = [[tmats_x[k], 0], [0, tmats_y[l]].
    centers_meas : list, shape (2, nbins)
        Coordinates of x and y bin centers on the screen.
    centers_rec : list, shape (2, nbins)
        Coordinates of x and y bin centers on the reconstruction grid.
    tmats_x{y} : list[ndarray], shape (n_proj,)
        List of 2 x 2 transfer matrices for x-x'{y-y'}.
    method : {'SART', 'FBP', 'MENT'}
        The 2D reconstruction method.
    proc_kws : dict
        Key word arguments for `process`.
    **kws
        Key word arguments for `rec2D`.
        
    Returns
    -------
    Z, ndarray, shape (n_bins, n_bins, n_bins, n_bins)
        Reconstructed phase space distribution. The grid coordinates are the 
        same for x-x' (centers_rec[0]) and for y-y' (centers_rec[1]).
    """        
    if proc_kws is None:
        proc_kws = dict()
    K = len(tmats_x)
    L = len(tmats_y)
    n_bins = n_bins = S.shape[0] # assume same number of x/y bins.
    xcenters_meas, ycenters_meas = centers_meas
    xcenters_rec, ycenters_rec = centers_rec
    
    D = np.zeros((n_bins, L, n_bins, n_bins))
    for j in trange(n_bins):
        for l in range(L):
            projections = S[:, j, :, l].T
            D[j, l, :, :] = rec2D(projections, tmats_x, 
                                  centers_meas[0], centers_rec[0],
                                  method=method, **kws)
    Z = np.zeros((n_bins, n_bins, n_bins, n_bins))
    for r in trange(n_bins):
        for s in range(n_bins):
            projections = D[:, :, r, s].T
            Z[r, s, :, :] = rec2D(projections, tmats_y, 
                                  centers_meas[1], centers_rec[1],
                                  method=method, **kws)
    Z = process(Z, **proc_kws)
    return Z


def art4D(projections, tmats, rec_centers, meas_centers):
    """Direct four-dimensional algebraic reconstruction (ART).
    
    We set up the linear system rho = P psi. Assume the x-x'-y-y' grid at the
    reconstruction grid has Nr**4 bins, the x-y grid on the screen has Ns**2
    bins, and that there are n measurements. Then rho is a vector with n*Ns**2
    elements of the measured density on the screen and psi is a vector with
    Nr**4 elements. P[i, j] = 1.0 if the jth bin center in the reconstruction
    grid ends up in the ith bin on the screen, or 0.0 otherwise. 
    
    P is a very sparse matrix. Currently, scipy.sparse.linalg.lsqr is used. A
    grid size of N = 50 has used successfuly, but N = 75 lead to an 'out of
    memory' error.
    
    Parameters
    ----------
    projections : list[ndarray, shape (Nsx, Nsy)]
        List of measured projections on the x-y plane.
    tmats : list[ndarray, shape (4, 4)]
        List of transfer matrices from the reconstruction location to the 
        measurement location.
    rec_centers : list[ndarray, shape (Nr,)]
        Grid center coordinates in [x, x', y, y'].
    meas_edges : list[ndarray, shape (Ns,)], shape (2,)
        Coordinates of bin centers on the screen in [x, y].
        
    Returns
    -------
    Z : ndarray, shape (Nr**4)
        Z[i, j, k, l] gives the phase space density at:
            x = rec_centers[0][i], 
            x' = rec_centers[1][j], 
            y = rec_centers[2][k], 
            y' = rec_centers[3][l].
    """
    # Treat each reconstruction bin center as a particle.
    print('Forming arrays.')
    rec_grid_coords = get_grid_coords(*rec_centers)
    n_bins_rec = [len(c) for c in rec_centers]
    rec_grid_size = np.prod(n_bins_rec)
    col_indices = np.arange(rec_grid_size)
    
    meas_edges = [get_bin_centers(_centers) for _centers in meas_centers]
    n_bins_meas = [len(_centers) for _centers in meas_centers]
    n_bins_meas[0] = len(meas_edges[0]) - 1
    n_bins_meas[1] = len(meas_edges[1]) - 1
    row_block_size = n_bins_meas[0] * n_bins_meas[1]
    n_proj = len(projections)
    rho = np.zeros(n_proj * row_block_size) # measured density on the screen.
    rows, cols = [], [] # nonzero row and column indices of P

    for proj_index in trange(n_proj):
        # Transport the reconstruction grid to the screen.
        M = tmats[proj_index]
        meas_grid_coords = np.apply_along_axis(lambda row: np.matmul(M, row), 1, rec_grid_coords)

        # For each particle, record the indices of the bin it landed in. So we
        # want (k, l) such that the particle landed in the bin with x = x[k] and
        # y = y[l] on the screen. One of the indices will be -1 or n_bins if the
        # particle landed outside the screen.
        xidx = np.digitize(meas_grid_coords[:, 0], meas_edges[0]) - 1
        yidx = np.digitize(meas_grid_coords[:, 2], meas_edges[1]) - 1
        on_meas = np.logical_and(
            np.logical_and(xidx >= 0, xidx < n_bins_meas[0]),
            np.logical_and(yidx >= 0, yidx < n_bins_meas[1])
        )
        # Get the indices for the flattened array.
        projection = projections[proj_index]
        meas_idx = np.ravel_multi_index((xidx, yidx), projection.shape, mode='clip')

        # P[i, j] = 1 if particle j landed in bin i on the screen, 0 otherwise.
        i_offset = proj_index * row_block_size
        for j in tqdm(col_indices[on_meas]):
            i = meas_idx[j] + i_offset
            rows.append(i)
            cols.append(j)
        rho[i_offset: i_offset + row_block_size] = projection.flat

    print('Creating sparse matrix P.')
    t = time.time()
    P = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), 
                          shape=(n_proj * row_block_size, rec_grid_size))
    print('Done. t = {}'.format(time.time() - t))

    print('Solving linear system.')
    t = time.time()
    psi, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = sparse.linalg.lsqr(P, rho, show=True, iter_lim=1000)
    print('Done. t = {}'.format(time.time() - t))

    print('Reshaping phase space density.')
    Z = psi.reshape(tuple(n_bins_rec))
    
    return Z

def pic4D(projections, tmats, rec_centers, meas_centers, max_iters=15):
    """Four-dimensional reconstruction using particle tracking.
    
    The method is described in Wang et al. (2019). It is not currently working.
    """
    n_dims = 4
    n_proj = len(projections)
    n_parts = 1000000
    rec_bin_widths = np.diff(rec_centers)[:, 0]
    rec_edges = [get_bin_edges(_centers) for _centers in rec_centers]
    rec_limits = [(min(_edges), max(_edges)) for _edges in rec_edges]
    projections_meas = np.copy(projections)
    meas_edges = [get_bin_edges(_centers) for _centers in meas_centers]
    meas_edges[0], meas_edges[1] = meas_edges
    
    # Generate initial coordinates uniformly within the reconstruction grid. 
    # The distribution should be large to ensure that a significant number of 
    # particles land on the screen.     
    mins = np.min(rec_limits, axis=1)
    maxs = np.max(rec_limits, axis=1)
    scale = 1.0
    lo = scale * mins
    hi = scale * maxs
    X = np.random.uniform(scale * mins, scale * maxs, size=(n_parts, n_dims))

    for iteration in range(max_iters):
        # Simulate the measurements.
        projections, coords_meas = [], []
        for M in tqdm(tmats):
            X_meas = apply(M, X)
            projection, _, _ = np.histogram2d(X_meas[:, 0], X_meas[:, 2], meas_edges)
            projection /= np.sum(projection)
            projections.append(projection)
            coords_meas.append(X_meas)
        projections = np.array(projections)
        coords_meas = np.array(coords_meas)

        # Weight particles.
        weights = np.zeros((n_proj, X.shape[0]))
        for k, X_meas in enumerate(coords_meas):
            xidx = np.digitize(X_meas[:, 0], meas_edges[0]) - 1
            yidx = np.digitize(X_meas[:, 2], meas_edges[1]) - 1
            on_meas_x = np.logical_and(xidx >= 0, xidx < len(meas_edges[0]) - 1)
            on_meas_y = np.logical_and(yidx >= 0, yidx < len(meas_edges[1]) - 1)
            on_meas = np.logical_and(on_meas_x, on_meas_y)
            weights[k, on_meas] = projections_meas[k, xidx[on_meas], yidx[on_meas]] 
            weights[k, on_meas] /= projections[k, xidx[on_meas], yidx[on_meas]]

        # Only keep particles that hit every meas.
        keep_idx = [np.all(weights[:, i] > 0.) for i in range(weights.shape[1])]
        weights[:, np.logical_not(keep_idx)] = 0.
        weights = np.sum(weights, axis=0)    
        weights /= np.sum(weights)

        # Convert the weights to counts.
        counts = weights * n_parts
        counts = np.round(counts).astype(int)
        
        # Generate a new bunch.
        add_idx = counts > 0
        lo = np.repeat(X[add_idx] - 0.5 * rec_bin_widths, counts[add_idx], axis=0)
        hi = np.repeat(X[add_idx] + 0.5 * rec_bin_widths, counts[add_idx], axis=0)
        X = np.random.uniform(lo, hi)
        
        proj_error = np.sum((projections_meas - projections)**2)
        print('proj_error = {}'.format(proj_error))
        print('New bunch has {} particles'.format(X.shape[0]))
        print('Iteration {} complete'.format(iteration))
        
        

        Z, _ = np.histogramdd(X, rec_edges)
        Z /= np.sum(Z)
        
        plot_kws = dict(ec='None', cmap='mono_r')
        labels = ["x", "x'", "y", "y'"]
        indices = [(0, 1), (2, 3), (0, 2), (0, 3), (2, 1), (1, 3)]
        fig, axes = pplt.subplots(nrows=1, ncols=6, figwidth=8.5, 
                                  sharex=False, sharey=False, space=0.2)
        for ax, (i, j) in zip(axes, indices):
            _Z = project(Z, [i, j])
            ax.pcolormesh(rec_edges[i], rec_edges[j], _Z.T, **plot_kws)
            ax.annotate('{}-{}'.format(labels[i], labels[j]),
                        xy=(0.02, 0.92), xycoords='axes fraction', 
                        color='white', fontsize='medium')
        axes.format(xticks=[], yticks=[])
        plt.show()
        
    Z = np.histogramdd(X, rec_edges)
    return Z, projections