"""
This module defines functions to integrate envelope equations.
"""
import numpy as np
from scipy.integrate import odeint
import envelope_analysis as ea


def chernin_derivs(s, v, Q, focusing_strength, deriv_calc='matrix'):
    """Compute derivative of 10 element moment vector.
    
    Originally derived in [1]. Here the notation of [2] is followed since they
    explicitly write out the equations. I haven't found any issues from using 
    matrix multiplication thought. 
    
    Inputs
    ------
    s (float)
        Longitudinal position in lattice [m].
    v (array-like)
        10 element moment vector:
        [s11, s12, s13, s14, s22, s23, s24, s33, s34, s44], where sij is the
        i,j element of the covariance matrix.
    perveance (float)
        Space charge perveance.
    focusing_strength (function)
        Function which returns the horizontal focusing strength at position s.
        Call signature is: k0x = focusing_strength(s).
    deriv_calc (str)
        Method to compute derivatives. 'matrix' uses matrix multiplication to 
        compute derivatives. 'vector' uses the 10 hard-coded equations. 
        
    Returns
    -------
    w : NumPy array
        Derivative of v with respect to s.
        
    References
    ----------
    [1] D. Chernin, Part. Accel. 24, 29 (1988).
    [2] A. Goswami, P. Sing Babu, V.S. Panditc, Eur. Phys. J. Plus 131, 393 
        (2016).
    """
    # Focusing strength in lattice (assume no skew elements)
    k0x = focusing_strength(s)
    k0y = -k0x
    k0xy = 0.0
    # Space charge terms
    s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = v
    S0 = np.sqrt(s11*s33 - s13**2)
    Sx = s11 + S0 
    Sy = s33 + S0 
    D = S0 * (Sx + Sy)
    psi_xx = Sy / D
    psi_yy = Sx / D
    psi_xy = -s13 / D
    # Modified focusing strength
    kx = (Q/2)*psi_xx - k0x
    ky = (Q/2)*psi_yy - k0y
    kxy = (Q/2)*psi_xy + k0xy
    # Derivatives
    if deriv_calc == 'matrix':
        M = np.array([[0, 1, 0, 0],
                      [kx, 0, kxy, 0],
                      [0, 0, 0, 1],
                      [kxy, 0, ky, 0]])
        Sigma = vec_to_mat(v)
        Sigma_prime = np.matmul(M, Sigma) + np.matmul(Sigma, M.T)
        w = mat_to_vec(Sigma_prime)
    elif deriv_calc == 'vector':
        w = np.zeros(10)
        w[0] = 2 * s12
        w[1] = s22 + kx*s11 + kxy*s13
        w[2] = s23 + s14
        w[3] = s24 + kxy*s11 + ky*s13
        w[4] = 2*kx*s12 + 2*kxy*s23
        w[5] = s24 + kx*s13 + kxy*s33
        w[6] = kx*s14 + kxy*(s34+s12) + ky*s23
        w[7] = 2 * s34
        w[8] = s44 + kxy*s13 + ky*s33
        w[9] = 2*kxy*s14 + 2*ky*s34
    return w


def tilted_envelope_derivs(s, v, perveance, focusing_strength):
    """Compute derivative of envelope parameter vector.
    
    Inputs
    ------
    s : float
        Longitudinal position in lattice [m].
    v : array-like
        Envelope parameter vector [a, b, a', b', e, f, e', f'].
    perveance :float
        Space charge perveance.
    focusing_strength : callable
        Function which returns the horizontal focusing strength at position s.
        Call signature is: k0x = focusing_strength(s).
        
    Returns
    -------
    w : NumPy array
        Derivative of v with respect to s.
    """
    a, b, ap, bp, e, f, ep, fp = v
        
    phi = ea.tilt_angle(v)
    cos, sin = np.cos(phi), np.sin(phi)
    cos2, sin2 = cos**2, sin**2
    a2b2, e2f2 = a**2 + b**2, e**2 + f**2
    
    cx, cy = ea.cxcy(v)
    T = 2 * perveance / (cx + cy)
    
    k0x = focusing_strength(s)
    k0y = -k0x
    
    w = np.zeros(8)
    w[0] = ap
    w[1] = bp
    w[2] = -k0x*a + T*((a*cos2 - e*sin*cos)/cx + (a*sin2 + e*sin*cos)/cy)
    w[3] = -k0x*b + T*((e*sin2 - a*sin*cos)/cx + (e*cos2 + a*sin*cos)/cy)
    w[4] = ep
    w[5] = fp
    w[6] = -k0y*e + T*((b*cos2 - f*sin*cos)/cx + (b*sin2 + f*sin*cos)/cy)
    w[7] = -k0y*f + T*((f*sin2 - b*sin*cos)/cx + (f*cos2 + b*sin*cos)/cy)
    return w
    

def mat_to_vec(S):
    """Return vector of independent elements in 4x4 symmetric matrix S."""   
    return np.array([S[0,0], S[0,1], S[0,2], S[0,3], S[1,1], 
                     S[1,2], S[1,3], S[2,2], S[2,3], S[3,3]])

def vec_to_mat(v):
    """Return symmetric matrix from vector."""
    S11, S12, S13, S14, S22, S23, S24, S33, S34, S44 = v
    return np.array([[S11, S12, S13, S14],
                     [S12, S22, S23, S24],
                     [S13, S23, S33, S34],
                     [S14, S24, S34, S44]])

def get_perveance(energy, mass, density):
    """Space charge perveance."""
    gamma = 1 + (energy / mass) # Lorentz factor           
    beta = np.sqrt(1 - (1 / (gamma**2))) # v/c   
    r0 = 1.53469e-18 # classical proton radius [m]
    return (2 * r0 * density) / (beta**2 * gamma**3)

def k_fodo(s):
    """Return focusing strength in FODO lattice of length 5.0 m (tune=0.136)."""
    k0 = 0.7122835574
    if s < 0.25 or s >= 4.75:
        return +k0
    elif 2.25 <= s < 2.75:
        return -k0
    else:
        return 0.0
