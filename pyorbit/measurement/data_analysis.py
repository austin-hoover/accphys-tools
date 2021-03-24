import numpy as np
import scipy.optimize as opt


def to_mat(sigma):
    """Return covariance matrix from 10 element moment vector."""
    s11, s22, s12, s33, s44, s34, s13, s23, s14, s24 = sigma
    return np.array([[s11, s12, s13, s14], 
                     [s12, s22, s23, s24], 
                     [s13, s23, s33, s34], 
                     [s14, s24, s34, s44]])

def to_vec(Sigma):
    """Return 10 element moment vector from covariance matrix."""
    s11, s12, s13, s14 = Sigma[0, :]
    s22, s23, s24 = Sigma[1, 1:]
    s33, s34 = Sigma[2, 2:]
    s44 = Sigma[3, 3]
    return np.array([s11, s22, s12, s33, s44, s34, s13, s23, s14, s24])
    
    
def reconstruct_moments(A, b):
    """Reconstruct covariance matrix from measured data.
    
    Solves the problem A.sigma = b, where sigma is the vector
    of 10 beam moments at the reconstruction point and A and 
    b are defined below.
    
    Parameters
    ----------
    A : ndarray, shape (3n, 10)
        Coefficient array determined by transfer matrix elements.
    b : ndarray, shape (3n,)
        Observation array determined by beam moments at measurement location.

    Returns
    -------
    ndarray, shape (4, 4)
        Covariance matrix at reconstruction point.
    """
    # Squared moments can't be negative
    lb = 10 * [-np.inf]
    lb[0] = lb[1] = lb[3] = lb[4] = 0.0 
    bounds = opt.Bounds(lb, np.inf, keep_feasible=False)

    result = opt.lsq_linear(A, b, bounds=(lb, np.inf))
    return to_mat(result.x)
