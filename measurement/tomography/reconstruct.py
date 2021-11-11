"""Image reconstuction.

All angles should be kept in radians. We convert to degrees only when passing the 
angles to skimage.
"""
import time

from tqdm import trange
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.interpolate import griddata
from skimage.transform import iradon
from skimage.transform import iradon_sart


def get_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def get_edges(centers):
    width = np.diff(centers)[0]
    return np.hstack([centers - 0.5 * width, [centers[-1] + 0.5 * width]])


def project(Z, indices):
    if type(indices) is int:
        indices = [indices]
    axis = tuple([k for k in range(Z.ndim) if k not in indices])
    return np.sum(Z, axis=axis)


def normalize(Z, bin_volume=1.0):
    Zn = np.copy(Z)
    A = np.sum(Zn)
    if A == 0.0:
        return Zn
    return Zn / A / bin_volume


def get_bin_volume(limits, n_bins):
    if type(n_bins) is int:
        n_bins = len(limits) * [n_bins]
    return np.prod([(np.diff(lim)[0] / n) for lim, n in zip(limits, n_bins)])


def process(Z, keep_positive=False, density=False, limits=None):
    if keep_positive:
        Z = np.clip(Z, 0.0, None)
    if density:
        bin_volume = 1.0 
        if limits is not None:
            bin_volume = get_bin_volume(limits, Z.shape)
        Z = normalize(Z, bin_volume)
    return Z
    
    
def fbp(projections, angles, keep_positive=False, density=False,
        limits=None, **kws):
    """Filtered Back Projection (FBP)."""
    n_bins, n_proj = projections.shape
    angles = np.degrees(angles)
    Z = iradon(projections, theta=-angles, **kws).T
    return process(Z, keep_positive, density, limits)


def sart(projections, angles, iterations=1, keep_positive=False,
         density=False, limits=None, **kws):
    """Simultaneous Algebraic Reconstruction (SART)"""
    angles = np.degrees(angles)
    Z = iradon_sart(projections, theta=-angles, **kws).T
    for _ in range(iterations - 1):
        Z = iradon_sart(projections, theta=-angles, image=Z.T, **kws).T
    return process(Z, keep_positive, density, limits)


def ment(projections, angles, **kws):
    """Maximum Entropy (MENT)."""
    raise NotImplementedError


def rec4D(S, muxx, muyy, method='SART', keep_positive=False, 
          density=False, limits=None, **kws):
    """4D reconstruction using method from Hock (2013).
    
    Parameters
    ----------
    """
    rfunc = None
    if method == 'SART':
        rfunc = sart
    elif method == 'FBP':
        rfunc = fbp
    elif method == 'MENT':
        rfunc = ment
    else:
        raise ValueError("Invalid method!")
    
    K = len(muxx)
    L = len(muyy)
    n_bins = n_bins = S.shape[0]
           
    D = np.zeros((n_bins, L, n_bins, n_bins))
    for j in trange(n_bins):
        for l in range(L):
            D[j, l, :, :] = rfunc(S[:, j, :, l], muxx, **kws)

    Z = np.zeros((n_bins, n_bins, n_bins, n_bins))
    for r in trange(n_bins):
        for s in range(n_bins):
            Z[r, s, :, :] = rfunc(D[:, :, r, s], muyy, **kws)
            
    return process(Z, keep_positive, density, limits)


def get_grid_coords(*xi, indexing='ij'):
    """Return array of shape (N, D), where N is the number of points on 
    the grid and D is the number of dimensions."""
    return np.vstack([X.ravel() for X in np.meshgrid(*xi, indexing='ij')]).T


def transform(Z, V, grid, new_grid=None):
    """Apply a linear transformation to a distribution.
    
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
        transformed space.
        
    Returns
    -------
    Z : ndarray, shape (len(x1), ..., len(xn))
        The distribution function in the original space. Linear interpolation
        is used to fill in the gaps.
    new_grid : list[array_like] (optional)
        List of 1D arrays [x1, x2, ...] representing the bin centers in the 
        transformed space.
    """        
    # Transform the grid coordinates.
    coords = get_grid_coords(*grid)
    coords_new = np.apply_along_axis(lambda row: np.matmul(V, row), 1, coords)
        
    # Define the interpolation coordinates.
    if new_grid is None:
        mins = np.min(coords_new, axis=0)
        maxs = np.max(coords_new, axis=0)
        new_grid = [np.linspace(mins[i], maxs[i], Z.shape[i]) for i in range(len(mins))]    
    coords_int = get_grid_coords(*new_grid)
    
    # Interpolate.
    Z = griddata(coords_new, Z.ravel(), coords_int, method='linear')
    Z[np.isnan(Z)] = 0.0
    Z = Z.reshape([len(xi) for xi in new_grid])
    return Z, new_grid



def art4D(projections, tmats, rec_grid_centers, screen_edges_x, screen_edges_y):
    """Direct four-dimensional algebraic reconstruction (ART).
    
    We set up the linear system rho = P psi. Assume the x-x'-y-y' grid at the reconstruction
    grid has Nr**4 bins, the x-y grid on the screen has Ns**2 bins, and that there are n
    measurements. Then rho is a vector with n*Ns**2 elements of the measured density on the
    screen and psi is a vector with Nr**4 elements. P[i, j] = 1.0 if the jth bin center in 
    the reconstruction grid ends up in the ith bin on the screen, or 0.0 otherwise. 
    
    P is a very sparse matrix. Currently, scipy.sparse.linalg.lsqr is used. A grid size of
    N = 50 has used successfuly, but N = 75 lead to an 'out of memory' error.
    
    Parameters
    ----------
    projections : list[ndarray, shape (Nsx, Nsy)]
        List of measured projections on the x-y plane.
    tmats : list[ndarray, shape (4, 4)]
        List of transfer matrices from the reconstruction location to the measurement location.
    rec_grid_centers : list[ndarray, shape (Nr,)]
        Grid center coordinates in [x, x', y, y'].
    screen_edges_x, screen_edges_y : ndarray, shape (Ns,)
        Coordinates of bin edges on the screen in [x, y].
        
    Returns
    -------
    Z : ndarray, shape (Nr**4)
        Z[i, j, k, l] gives the phase space density at 
        x = rec_grid_centers[0][i], 
        x' = rec_grid_centers[1][j], 
        y = rec_grid_centers[2][k], 
        y' = rec_grid_centers[3][l].
    """
    
    print('Forming arrays.')

    # Treat each reconstruction bin center as a particle. 
    rec_grid_coords = get_grid_coords(*rec_grid_centers)
    n_bins_rec = [len(c) for c in rec_grid_centers]
    rec_grid_size = np.prod(n_bins_rec)
    col_indices = np.arange(rec_grid_size)

    n_bins_x_screen = len(screen_edges_x) - 1
    n_bins_y_screen = len(screen_edges_y) - 1
    row_block_size = n_bins_x_screen * n_bins_y_screen
    n_proj = len(projections)
    rho = np.zeros(n_proj * row_block_size) # measured density on the screen.
    rows, cols = [], [] # nonzero row and column indices of P

    for proj_index in trange(n_proj):
        # Transport the reconstruction grid to the screen.
        M = tmats[proj_index]
        screen_grid_coords = np.apply_along_axis(lambda row: np.matmul(M, row), 1, rec_grid_coords)

        # For each particle, record the indices of the bin it landed in. So we want (k, l) such
        # that the particle landed in the bin with x = x[k] and y = y[l] on the screen. One of 
        # the indices will be -1 or n_bins if the particle landed outside the screen.
        xidx = np.digitize(screen_grid_coords[:, 0], screen_edges_x) - 1
        yidx = np.digitize(screen_grid_coords[:, 2], screen_edges_y) - 1
        on_screen = np.logical_and(np.logical_and(xidx >= 0, xidx < n_bins_x_screen), 
                                   np.logical_and(yidx >= 0, yidx < n_bins_y_screen))

        # Get the indices for the flattened array.
        projection = projections[proj_index]
        screen_idx = np.ravel_multi_index((xidx, yidx), projection.shape, mode='clip')

        # P[i, j] = 1 if particle j landed in bin i on the screen, 0 otherwise.
        i_offset = proj_index * row_block_size
        for j in tqdm(col_indices[on_screen]):
            i = screen_idx[j] + i_offset
            rows.append(i)
            cols.append(j)
        rho[i_offset: i_offset + row_block_size] = projection.flat


    print('Creating sparse matrix P.')
    t = time.time()
    P = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_proj * row_block_size, rec_grid_size))
    print('Done. t = {}'.format(time.time() - t))

    print('Solving linear system.')
    t = time.time()
    psi, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = sparse.linalg.lsqr(P, rho, show=True, iter_lim=1000)
    print()
    print('Done. t = {}'.format(time.time() - t))

    print('Reshaping phase space density.')
    Z = psi.reshape(tuple(n_bins_rec))
    
    return Z