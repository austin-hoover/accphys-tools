from __future__ import print_function
import sys
import os

import numpy as np
import scipy.optimize as opt


def ancestor_folder_path(current_path, ancestor_folder_name):  
    parent_path = os.path.dirname(current_path)
    if parent_path == current_path:
        raise ValueError("Couldn't find ancestor folder.")
    if parent_path.split('/')[-1] == ancestor_folder_name:
        return parent_path
    return ancestor_folder_path(parent_path, ancestor_folder_name)

sys.path.append(ancestor_folder_path(os.path.abspath(__file__), 'accphys'))
from tools import beam_analysis as ba
from tools import utils



def is_physical_cov(Sigma):
    """Return True if the covariance matrix is physical."""
    if not utils.is_positive_definite(Sigma) or np.linalg.det(Sigma) < 0.:
        return False
    eps_x, eps_y, eps_1, eps_2 = ba.emittances(Sigma)
    if (eps_x * eps_y < eps_1 * eps_2):
        return False
    return True


def to_mat(sigma):
    """Return covariance matrix from 10 element moment vector.
    
    The ordering follows Prat (2014).
    """
    s11, s22, s12, s33, s44, s34, s13, s23, s14, s24 = sigma
    return np.array([[s11, s12, s13, s14], 
                     [s12, s22, s23, s24], 
                     [s13, s23, s33, s34], 
                     [s14, s24, s34, s44]])

def to_vec(Sigma):
    """Return 10 element moment vector from covariance matrix.
    
    The ordering follows Prat (2014).
    """
    s11, s12, s13, s14 = Sigma[0, :]
    s22, s23, s24 = Sigma[1, 1:]
    s33, s34 = Sigma[2, 2:]
    s44 = Sigma[3, 3]
    return np.array([s11, s22, s12, s33, s44, s34, s13, s23, s14, s24])
    
    
def reconstruct(transfer_mats, moments, **lsq_kws):
    """Reconstruct the covariance matrix.
    
    Parameters
    ----------
    transfer_mats : list of (4, 4) ndarray, length n
        List of 4x4 transfer matrix at each scan.
    moments : list or ndarray, shape (n, 3)
        List of [<xx>, <yy>, <xy>] moments for each scan.
    **lsq_kws
        Key word arguments passed to scipy.optimize.lsq_linear

    Returns
    -------
    ndarray, shape (4, 4)
        The reconstructed covariance matrix.
    """
    A, b = [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
        A.append([M[0, 0]**2, M[0, 1]**2, 2*M[0, 0]*M[0, 1], 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, M[2, 2]**2, M[2, 3]**2, 2*M[2, 2]*M[2, 3], 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, M[0, 0]*M[2, 2],  M[0, 1]*M[2, 2],  M[0, 0]*M[2, 3],  M[0, 1]*M[2, 3]])
        b.append(sig_xx)
        b.append(sig_yy)
        b.append(sig_xy)
    lb = 10 * [-np.inf]
    lb[0] = lb[1] = lb[3] = lb[4] = 0.0 # Squared moments can't be negative
    result = opt.lsq_linear(A, b, bounds=(lb, np.inf), **lsq_kws)
    return to_mat(result.x)


def propagate_emittance_errors(Sigma, C):
    """Compute standard deviation of beam covariance matrix from computed Sigma 
    and LLSQ covariance matrix C."""
    eps_x, eps_y, eps_1, eps_2 = ba.emittances(Sigma)
    
    Cxx = C[:3, :3]
    Cyy = C[3:6, 3:6]
    Cxy = C[6:, 6:]
    
    grad_eps_x = (0.5 / eps_x) * np.array([Sigma[1, 1], Sigma[0, 0], -2 * Sigma[0, 1]])
    grad_eps_y = (0.5 / eps_y) * np.array([Sigma[3, 3], Sigma[2, 2], -2 * Sigma[2, 3]])
    eps_x_std = np.sqrt(np.linalg.multi_dot([grad_eps_x.T, Cxx, grad_eps_x]))
    eps_y_std = np.sqrt(np.linalg.multi_dot([grad_eps_y.T, Cyy, grad_eps_y]))
    
    g1 = np.linalg.det(Sigma)
    U = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    SigmaU = np.matmul(Sigma, U)
    g2 = np.trace(np.matmul(SigmaU, SigmaU))
    g = np.array(g1, g2)
    
    s11, s12, s13, s14 = Sigma[0, :]
    s22, s23, s24 = Sigma[1, 1:]
    s33, s34 = Sigma[2, 2:]
    s44 = Sigma[3, 3]

    grad_g = np.zeros((10, 2))
    grad_g[0, 0] = -s33*s24**2 + 2*s23*s24*s34 - s44*s23**2 + s22*(s33*s44 - s34**2)
    grad_g[1, 0] = -s33*s14**2 + 2*s13*s14*s34 - s44*s13**2 + s11*(s33*s44 - s34**2)
    grad_g[2, 0] = 2 * (s14*(s24*s33 - s23*s34) + s13*(-s24*s34 + s23*s44) + s12*(s34**2 - s33*s44))
    grad_g[3, 0] = -s22*s14**2 + 2*s12*s14*s24 - s44*s12**2 + s11*(s22*s44 - s24**2)
    grad_g[4, 0] = -s22*s13**2 + 2*s12*s13*s23 - s33*s12**2 + s11*(s22*s33 - s23**2)
    grad_g[5, 0] = 2 * (-s12*s14*s23 + s13*(s14*s22 - s12*s24) + s34*s12**2 + s11*(s23*s24 - s22*s34))
    grad_g[6, 0] = 2 * (s14*(-s23*s24 + s22*s34) + s13*(s24**2 - s22*s44) + s12*(-s24*s34 + s23*s44))
    grad_g[7, 0] = 2 * (s23*s14**2 - s14*(s13*s24 + s12*s34) + s12*s13*s44 + s11*(s24*s34 - s23*s44))
    grad_g[8, 0] = 2 * (s14*(s23**2 - s22*s33) + s13*(-s23*s24 + s22*s34) + s12*(s24*s33 - s23*s34))
    grad_g[9, 0] = 2 * (s24*s13**2 + s12*s14*s33 - s13*(s14*s23 + s12*s34) + s11*(-s24*s33 + s23*s34))

    grad_g[0, 1] = -2 * s22
    grad_g[1, 1] = -2 * s11
    grad_g[2, 1] = 4 * s12
    grad_g[3, 1] = -2 * s44
    grad_g[4, 1] = -2 * s33
    grad_g[5, 1] = 4 * s34
    grad_g[6, 1] = -4 * s24
    grad_g[7, 1] = 4 * s24
    grad_g[8, 1] = 4 * s23
    grad_g[9, 1] = -4 * s13
    
    Cg = np.linalg.multi_dot([grad_g.T, C, grad_g])
    
    H = np.sqrt(g2**2 - 16 * g1)
    deps1_dg1 = -1 / (eps_1 * H)
    deps2_dg1 = -1 / (eps_2 * H)
    deps1_dg2 = (1/8) * (1/eps_1) * (+g2/H - 1)
    deps2_dg2 = (1/8) * (1/eps_2) * (-g2/H - 1)
    grad_eps_1 = np.array([deps1_dg1, deps1_dg2])
    grad_eps_2 = np.array([deps2_dg1, deps2_dg2])
    
    eps_1_std = np.linalg.multi_dot([grad_eps_1.T, Cg, grad_eps_1])
    eps_2_std = np.linalg.multi_dot([grad_eps_2.T, Cg, grad_eps_2])

    return np.array([eps_x_std, eps_y_std, eps_1_std, eps_2_std])


def propagate_twiss_errors(Sigma, C):
    """Compute standard deviation of Twiss parameters from computed Sigma and 
    LLSQ covariance matrix C."""
    eps_x, eps_y, eps_1, eps_2 = ba.emittances(Sigma)
    alpha_x, alpha_y, beta_x, beta_y = ba.twiss2D(Sigma)
    Cxx = C[:3, :3]
    Cyy = C[3:6, 3:6]
    sig_xx_std, sig_xpxp_std, sig_xxp_std = np.sqrt(Cxx.diagonal())
    sig_yy_std, sig_ypyp_std, sig_yyp_std = np.sqrt(Cyy.diagonal())
    eps_x_std, eps_y_std, eps_1_std, eps_2_std = propagate_emittance_errors(Sigma, C)
    beta_x_std = np.sqrt((sig_xx_std / eps_x)**2 + (Sigma[0, 0] * eps_x_std / eps_x**2)**2)
    beta_y_std = np.sqrt((sig_yy_std / eps_y)**2 + (Sigma[2, 2] * eps_y_std / eps_y**2)**2)
    alpha_x_std = np.sqrt((sig_xxp_std / eps_x)**2 + (Sigma[0, 1] * eps_x_std / eps_x**2)**2)
    alpha_y_std = np.sqrt((sig_yyp_std / eps_y)**2 + (Sigma[2, 3] * eps_y_std / eps_y**2)**2)
    return np.array([alpha_x_std, alpha_y_std, beta_x_std, beta_y_std])


def llsq_cov_mat(A, residuals):
    """Compute the LLSQ covariance matrix from the coefficient matrix A 
    (from Ax = b) and the vector of residuals."""
    n, p = np.shape(A)
    if n == p:
        C = np.linalg.inv(A)
    else:
        C = (residuals / (n - p)) * np.linalg.inv(np.matmul(A.T, A))
    return C


def reconstruct(transfer_matrices, moments):
    """Reconstruct the covariance matrix.
    
    Parameters
    ----------
    transfer_matrices : list or ndarray, shape (n, 4, 4)
        Transfer matrices from the reconstruction location to the measurement locations.
    moments : list or ndarray, shape (n, 3)
        The [<xx>, <yy>, <xy>] moments.

    Returns
    -------
    Sigma : ndarray, shape (4, 4)
        The reconstructed covariance matrix.
    C : ndarray, shape (10, 10)
        The LLSQ covariance matrix.
    """
    # Form coefficient arrays and target arrays, 
    Axx, Ayy, Axy, bxx, byy, bxy = [], [], [], [], [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_matrices, moments):
        Axx.append([M[0, 0]**2, M[0, 1]**2, 2*M[0, 0]*M[0, 1]])
        Ayy.append([M[2, 2]**2, M[2, 3]**2, 2*M[2, 2]*M[2, 3]])
        Axy.append([M[0, 0]*M[2, 2],  M[0, 1]*M[2, 2],  M[0, 0]*M[2, 3],  M[0, 1]*M[2, 3]])
        bxx.append(sig_xx)
        byy.append(sig_yy)
        bxy.append(sig_xy)
    Axx = np.array(Axx)
    Ayy = np.array(Ayy)
    Axy = np.array(Axy)

    # Solve LLSQ problem.
    vec_xx, res_xx, _, _ = np.linalg.lstsq(Axx, bxx, rcond=None)
    vec_yy, res_yy, _, _ = np.linalg.lstsq(Ayy, byy, rcond=None)
    vec_xy, res_xy, _, _ = np.linalg.lstsq(Axy, bxy, rcond=None)
    
    # Form beam covariance matrix.
    sig_11, sig_22, sig_12 = vec_xx
    sig_33, sig_44, sig_34 = vec_yy
    sig_13, sig_23, sig_14, sig_24 = vec_xy
    Sigma = np.array([[sig_11, sig_12, sig_13, sig_14],
                      [sig_12, sig_22, sig_23, sig_24],
                      [sig_13, sig_23, sig_33, sig_34],
                      [sig_14, sig_24, sig_34, sig_44]])

    # Compute standard deviation of parameter vectors.
    if len(res_xy) == 0:
        res_xy = 1e-8
    C = np.zeros((10, 10))
    C[:3, :3] = llsq_cov_mat(Axx, float(res_xx))
    C[3:6, 3:6] = llsq_cov_mat(Ayy, float(res_yy))
    C[6:, 6:] = llsq_cov_mat(Axy, float(res_xy)) 
    return Sigma, C


def get_eps4D_std(eps_1, eps_2, eps_1_std, eps_2_std):
    return np.sqrt((eps_2 * eps_1_std)**2 + (eps_1 * eps_2_std)**2)