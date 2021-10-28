"""Image reconstuction.

All angles should be kept in radians. We convert to degrees only when passing the 
angles to skimage.
"""
from tqdm import trange
from tqdm import tqdm
import numpy as np
from scipy.interpolate import griddata
from skimage.transform import iradon
from skimage.transform import iradon_sart


def project(Z, indices):
    if type(indices) is int:
        indices = [indices]
    axis = tuple([k for k in range(4) if k not in indices])
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