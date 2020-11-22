"""
This module provides functions to analyze the 6D phase space coordinate data
describing a distribution of particles.
"""

# 3rd party
import numpy as np
import numpy.linalg as la
import pandas as pd
from tqdm import trange, tqdm

# My modules
from .utils import mat2vec, vec2mat, cov2corr, norm_mat
    
# Module level variables
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss_cols = ['ax','ay','bx','by','ex','ey','e1','e2']
dims = ['x','xp','y','yp','z','dE']

#------------------------------------------------------------------------------

def rms_ellipse_params(Sigma):
    s11, s33, s13 = Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]
    phi = 0.5 * np.arctan2(2 * s13, s11 - s33)
    cx = np.sqrt(2) * np.sqrt(s11 + s33 + np.sqrt((s11 - s33)**2 + 4*s13**2))
    cy = np.sqrt(2) * np.sqrt(s11 + s33 - np.sqrt((s11 - s33)**2 + 4*s13**2))
    return phi, cx, cy
    
    
def mode_emittances(Sigma):
    # Get imaginary components of eigenvalues of S.U
    U = np.array([[0,1,0,0], [-1,0,0,0], [0,0,0,1], [0,0,-1,0]])
    eigvals = la.eigvals(np.matmul(Sigma, U)).imag
    # Keep positive values
    eigvals = eigvals[np.argwhere(eigvals >= 0).flatten()]
    # If one of the mode emittances is zero, both will be kept.
    # Remove the extra zero.
    if len(eigvals) > 2:
        eigvals = eigvals[:-1]
    # Return the largest emittance first
    e1, e2 = np.sort(eigvals)
    return e1, e2
    
    
def get_twiss(Sigma):
    ex = np.sqrt(la.det(Sigma[:2, :2]))
    ey = np.sqrt(la.det(Sigma[2:, 2:]))
    bx = Sigma[0, 0] / ex
    by = Sigma[2, 2] / ey
    ax = -Sigma[0, 1] / ex
    ay = -Sigma[2, 3] / ey
    e1, e2 = mode_emittances(Sigma)
    return np.array([ax, ay, bx, by, ex, ey, e1, e2])
    
    
def read_moments(filename):
    """Read the bunch moments.
    
    Other beam parameters, such as the Twiss parameters and emittance, may be
    computed from the moments.
    
    Inputs
    ------
    filename : str
        The name of the file containing the transverse beam moments. The
        columns are ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2',
        'yyp','yp2'], where x2 = <x^2>, xxp = <xx'>, etc.
        
    Returns
    ------
    Stats object
        Object with the following DataFrames as fields:
        * twiss : the Twiss parameters and emittances
        * moments : the 10 elements of the covariance matrix
        * corr : the 10 elements of the correlation matrix
        * beam : the tilt angle, radii and area of the real space ellipse
    """
    moments = pd.read_table(filename, sep=' ', names=moment_cols)
        
    nframes = moments.shape[0]
    corr = np.zeros((nframes, 10))
    twiss = np.zeros((nframes, 8))
    beam = np.zeros((nframes, 4))
    
    for i, moment_vec in enumerate(moments.values):
        cov_mat = vec2mat(moment_vec)
        corr_mat = cov2corr(cov_mat)
        corr[i] = mat2vec(corr_mat)
        twiss[i] = get_twiss(cov_mat)
        phi, cx, cy = rms_ellipse_params(cov_mat)
        area = np.pi * cx * cy
        beam[i] = [np.degrees(phi), cx, cy, area]
        
    # Create DataFrames
    moments = pd.DataFrame(moments, columns=moment_cols)
    moments[['x_rms','y_rms']] = np.sqrt(moments[['x2','y2']])
    moments[['xp_rms','yp_rms']] = np.sqrt(moments[['xp2','yp2']])
    corr = pd.DataFrame(corr, columns=moment_cols)
    twiss = pd.DataFrame(twiss, columns=twiss_cols)
    twiss['e4D'] = twiss['e1'] * twiss['e2']
    beam = pd.DataFrame(beam, columns=['phi','cx','cy','area'])
    beam['area_rel'] = beam['area'] / beam.loc[0, 'area']
    
    class Stats:
        """Container for beam statistics. Each attribute is a DataFrame."""
        def __init__(self, twiss, moments, corr, beam):
            self.twiss = twiss
            self.moments = moments
            self.corr = corr
            self.beam = beam
            self.dfs = [self.twiss, self.moments, self.corr, self.beam]
    
    return Stats(twiss, moments, corr, beam)
