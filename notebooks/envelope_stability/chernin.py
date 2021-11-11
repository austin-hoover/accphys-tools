import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint


def mat2vec(Sigma):
    """Return vector of independent elements in 4x4 symmetric matrix Sigma.
    
    The order is by row in the upper triangular elements: [<xx>, <xx'>,
    <xy>, <xy'>, <x'x'>, <yx'>, <x'y'>, <yy>, <yy'>, <y'y'>].
    """
    return Sigma[np.triu_indices(4)]


def get_shape_factors(sig11, sig33, sig13):
    """Get space charge factors for envelope equations."""
    S0 = np.sqrt(sig11 * sig33 - sig13**2)
    Sx = sig11 + S0
    Sy = sig33 + S0
    D = S0 * (Sx + Sy)
    return S0, Sx, Sy, D


def derivs(y, s, Q, ext_foc):
    """Compute derivative of 10 element moment vector.
    
    Originally derived in [1]. Equations are written explicitly in [2].
    
    Parameters
    ----------
    y : ndarray, shape (10,)
        10 element moment vector:
        [sig11, sig12, sig13, sig14, sig22, sig23, sig24, sig33, sig34, sig44], where sij is four times the 
        the i,j element of the covariance matrix.
    s : float
        Longitudinal position in lattice [m].
    Q : float
        Dimensionless space charge perveance.
    ext_foc : callable
        Function which returns the horizontal, vertical, and skew focusing
        strength at a given position. Call signature is: 
        `k0x, k0y, k0xy = ext_foc(s)`. 
        
    Returns
    -------
    y_prime : ndarray, shape (10,)
        Derivative of y with respect to s.
        
    References
    ----------
    [1] D. Chernin, Part. Accel. 24, 29 (1988).
    [2] A. Goswami, P. Sing Babu, V.S. Panditc, Eur. Phys. J. Plus 131, 393 
        (2016).
    """
    # Focusing strength in lattice
    k0xx, k0yy, k0xy = ext_foc(s)
    # Space charge terms
    sig11, sig12, sig13, sig14, sig22, sig23, sig24, sig33, sig34, sig44 = y
    S0, Sx, Sy, D = get_shape_factors(sig11, sig33, sig13)
    qxx, qyy, qxy = Sy/D, Sx/D, -sig13/D
    # Modified focusing strength
    kxx = k0xx - 2 * Q * qxx
    kyy = k0yy - 2 * Q * qyy
    kxy = k0xy + 2 * Q * qxy
    # Derivatives
    y_prime = np.zeros(10)
    y_prime[0] = 2 * sig12
    y_prime[1] = sig22 - kxx*sig11 + kxy*sig13
    y_prime[2] = sig23 + sig14
    y_prime[3] = sig24 + kxy*sig11 - kyy*sig13
    y_prime[4] = -2*kxx*sig12 + 2*kxy*sig23
    y_prime[5] = sig24 - kxx*sig13 + kxy*sig33
    y_prime[6] = -kxx*sig14 + kxy*(sig34+sig12) - kyy*sig23
    y_prime[7] = 2 * sig34
    y_prime[8] = sig44 + kxy*sig13 - kyy*sig33
    y_prime[9] = 2*kxy*sig14 - 2*kyy*sig34
    return y_prime


def perturbed_derivs(yp, s, y0, Q, ext_foc):
    """Return derivative of perturbed moment vector.
    
    Parameters
    ----------
    yp : ndarray, shape (10,)
        Vector of deviations from matched moments.
    s : float
        Longitudinal position in lattice [m].
    y0 : ndarray, shape (10,)
        Vector of matched moments.
    Q : float
        Dimensionless space charge perveance.
    ext_foc : callable
        Function which returns the horizontal, vertical, and skew focusing
        strength at a given position. Call signature is: 
        `k0x, k0y, k0xy = ext_foc(s)`. 
    """
    J = jacobian(yp, s, y0, Q, ext_foc)
    return np.matmul(J, yp)


def jacobian(yp, s, y0, Q, ext_foc):
    """Return Jacobian matrix."""
    sig11, sig12, sig13, sig14, sig22, sig23, sig24, sig33, sig34, sig44 = y0
    S0, Sx, Sy, D = get_shape_factors(sig11, sig33, sig13)
    qxx, qyy, qxy = Sy/D, Sx/D, -sig13/D

    kxx_ext, kyy_ext, kxy_ext = ext_foc(s)
    kxx0 = kxx_ext - 2 * Q * qxx
    kyy0 = kyy_ext - 2 * Q * qyy
    kxy0 = kxy_ext + 2 * Q * qxy
        
    dxx = (sig33 / (2 * S0**2)) + qxx
    dyy = (sig11 / (2 * S0**2)) + qyy
    dxy = (-sig13 / S0**2) + 2*qxy
    
    a1 = qxx * ((sig33 / (2 * S0 * Sy)) - dxx) # dqxx_dsig11
    b1 = qxx * ((-sig13 / (S0 * Sy)) - dxy) # dqxx_dsig13
    c1 = qxx * (((1 + sig11/(2*S0)) / Sy) - dyy) # dqxx_dsig33
    a2 = qyy * (((1 + sig33/(2*S0)) / Sx) - dxx) # dqyy_dsig11
    b2 = qyy * ((-sig13 / (S0 * Sx)) - dxy) # dqyy_dsig13
    c2 = qyy * ((sig11 / (2 * S0 * Sx)) - dyy) # dqyy_dsig33
    a3 = -qxy * dxx # dqxy_dsig11
    if sig13 != 0:
        b3 = qxy * ((1 / sig13) - dxy) # dqxy_dsig13
    else:
        b3 = 0.0
    c3 = -qxy * dyy # dqxy_dsig33
    
    J = np.zeros((10, 10))
    
    J[0, 1] = 2.0
    
    J[1, 0] = kxx0 + 2*Q*(a1*sig11 + a2*sig13)
    J[1, 2] = kxy0 + 2*Q*(b1*sig11 + b2*sig13)
    J[1, 4] = 1.0
    J[1, 7] = 2*Q*(c1*sig11 + c2*sig13)
    
    J[2, 3] = J[2, 5] = 1.0
    
    J[3, 0] = kxy0 + 2*Q*(a2*sig11 + a3*sig13)
    J[3, 2] = kyy0 + 2*Q*(b2*sig11 + b3*sig13)
    J[3, 6] = 1.0
    J[3, 7] = 2*Q*(c2*sig11 + c3*sig13)
    
    J[4, 0] = 4*Q*(a1*sig12 + a2*sig23)
    J[4, 1] = 2*kxx0
    J[4, 2] = 4*Q*(b1*sig12 + b2*sig23)
    J[4, 5] = 2*kxy0
    J[4, 7] = 4*Q*(c1*sig12 + c2*sig23)
    
    J[5, 0] = 2*Q*(a1*sig13 + a2*sig33)
    J[5, 2] = kxx0 + 2*Q*(b1*sig13 + b2*sig33)
    J[5, 6] = 1.0
    J[5, 7] = kxy0 + 2*Q*(c1*sig13 + c2*sig33)
    
    J[6, 0] = 2*Q*(a1*sig14 + a3*sig23 + a2*(sig12 + sig34))
    J[6, 1] = kxy0
    J[6, 2] = 2*Q*(b1*sig14 + b3*sig23 + b2*(sig12 + sig34))
    J[6, 3] = kxx0
    J[6, 5] = kyy0
    J[6, 7] = 2*Q*(c1*sig14 + c3*sig23 + c2*(sig12 + sig34))
    J[6, 8] = kxy0
    
    J[7, 8] = 2.0
    
    J[8, 0] = 2*Q*(a2*sig13 + a3*sig33)
    J[8, 2] = kxy0 + 2*Q*(b2*sig13 + b3*sig33)
    J[8, 7] = kyy0 + 2*Q*(c2*sig13 + c3*sig33)
    J[8, 9] = 1.0
    
    J[9, 0] = 4*Q*(a2*sig14 + a3*sig23)
    J[9, 2] = 4*Q*(b2*sig14 + b3*sig23)
    J[9, 3] = 2*kxy0
    J[9, 7] = 4*Q*(c2*sig14 + c3*sig23)
    J[9, 8] = 2*kyy0
    
    return J





def track(y0, Q, ext_foc, positions):
    moments = odeint(derivs, y0, positions, args=(Q, ext_foc), atol=1e-14)
    return to_df(moments, positions)


def track_perturbed(yp, y0, Q, ext_foc, positions):
    perturbed_moments = odeint(perturbed_derivs, yp, positions, args=(y0, Q, ext_foc), atol=1e-14)
    return to_df(perturbed_moments, positions)


def to_df(moments, positions):
    """Convert ndarray of moments to DataFrame."""
    columns = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
    df = 1e6 * pd.DataFrame(moments, columns=columns)
    df[['x_rms','y_rms']] = np.sqrt(df[['x2','y2']])
    df['s'] = positions
    return df

