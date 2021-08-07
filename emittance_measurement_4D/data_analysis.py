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
    
    
def reconstruct(transfer_mats, moments, **kwargs):
    """Reconstruct covariance matrix from wire-scanner data.
    
    Parameters
    ----------
    transfer_mats : list of (4, 4) ndarray, length n
        List of 4x4 transfer matrix at each scan.
    moments : list or ndarray, shape (n, 3)
        List of [<xx>, <yy>, <xy>] moments for each scan.
    **kwargs
        Key word arguments passed to scipy.optimize.lsq_linear

    Returns
    -------
    ndarray, shape (4, 4)
        Covariance matrix at reconstruction point.
    """
    A, b = [], []
    for M, (x2, y2, xy) in zip(transfer_mats, moments):
        A.append([M[0, 0]**2, M[0, 1]**2, 2*M[0, 0]*M[0, 1], 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, M[2, 2]**2, M[2, 3]**2, 2*M[2, 2]*M[2, 3], 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, M[0, 0]*M[2, 2],  M[0, 1]*M[2, 2],  M[0, 0]*M[2, 3],  M[0, 1]*M[2, 3]])
        b.append(x2)
        b.append(y2)
        b.append(xy)
    A, b = np.array(A), np.array(b)
    lb = 10 * [-np.inf]
    lb[0] = lb[1] = lb[3] = lb[4] = 0.0 # Squared moments can't be negative
    result = opt.lsq_linear(A, b, bounds=(lb, np.inf), **kwargs)
    return to_mat(result.x)
