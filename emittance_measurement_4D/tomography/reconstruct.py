"""Image reconstuction.

All angles should be kept in radians. We convert to degrees only when passing the 
angles to skimage.
"""
from tqdm import trange
from tqdm import tqdm
import numpy as np
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


def rec4D(S, muxx, muyy, n_bins, method='SART', keep_positive=False, 
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
    
    # We first reconstruct a 3D projection of the 4D phase space. Consider one
    # row of the beam image; the intensity along the row gives a 1D projection
    # onto the $x$ plane for a vertical slice of the distribution. We have a 
    # set of these 1D projections at different $\theta_k$, and we can use this 
    # set to reconstruct the $x$-$x'$ distribution at this vertical slice. 
    # This is then repeated at each slice to give $f(x, x', y) = 
    # \int{f(x, x', y, y'}dy'$. We store this as an array $D$ such that 
    # $D_{j,l,r,s}$ gives the density at $x = x_r$, $x' = x'_s$ for 
    # $y = y_j$ and $\theta_y = \pi l / K$.         
    D = np.zeros((n_bins, L, n_bins, n_bins))
    for j in trange(n_bins):
        for l in range(L):
            D[j, l, :, :] = rfunc(S[:, j, :, l], muxx, **kws)
            
    # We now do a similar thing in the vertical plane. Choose one bin in 
    # reconstructed x-x' grid — (x_r, x'_s). At this bin, there is a set of 
    # numbers that define a 1D projection onto the y axis. And we have one 
    # such projection for each theta_y_l. Thus, the y-y' distribution can be 
    # reconstruted at each bin in the reconstructed x-x' grid. We store this 
    # as an array Z such that Z_{r,s,t,u} gives the density at x_r, x'_s,
    # y_t, y'_u.
    Z = np.zeros((n_bins, n_bins, n_bins, n_bins))
    for r in trange(n_bins):
        for s in range(n_bins):
            Z[r, s, :, :] = rfunc(D[:, :, r, s], muyy, **kws)
            
    return process(Z, keep_positive, density, limits)