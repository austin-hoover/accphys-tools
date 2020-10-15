"""
This module provides functions to analyze the 6D phase space coordinate data
describing a distribution of particles.
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
from tqdm import trange, tqdm

from .utils import mat2vec, vec2mat, cov2corr, norm_mat_4D
    
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss_cols = ['ax','ay','bx','by','ex','ey','e1','e2']
dims = ['x','xp','y','yp','z','dE']


def rms_ellipse_params(moments):
    s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = moments
    phi = 0.5 * np.arctan2(2 * s13, s11 - s33)
    cx = np.sqrt(2) * np.sqrt(s11 + s33 + np.sqrt((s11 - s33)**2 + 4*s13**2))
    cy = np.sqrt(2) * np.sqrt(s11 + s33 - np.sqrt((s11 - s33)**2 + 4*s13**2))
    return phi, cx, cy


def get_mode_emittances(S):
    """Compute the mode emittances from the covariance matrix.
    
    S : NumPy array, shape (4, 4)
        The transverse covariance matrix.
    """
    
    # Get imaginary components of eigenvalues of S.U
    U = np.array([[0,1,0,0], [-1,0,0,0], [0,0,0,1], [0,0,-1,0]])
    eigvals = la.eigvals(np.matmul(S, U)).imag

    # Keep positive values
    eigvals = eigvals[np.argwhere(eigvals >= 0).flatten()]

    # If one of the mode emittances is zero, both will be kept.
    # Remove the extra zero.
    if len(eigvals) > 2:
        eigvals = eigvals[:-1]
        
    # Return the largest emittance first
    eigvals = np.sort(eigvals)
    e1, e2 = np.sort(eigvals)
    return e1, e2


def get_twiss(S):
    """Compute the transverse Twiss parameters.

    Parameters
    ----------
    S : NumPy array, shape (4, 4)
        The transverse covariance matrix.
        
    Returns
    -------
    ax{y} : float
        The x{y} alpha parameter.
    bx{y} : float
        The x{y} beta parameter.
    ex{y} : float
        The x{y} rms emittance.
    e1{2} : float
        The mode emittance.
    """
    ex = np.sqrt(la.det(S[:2, :2]))
    ey = np.sqrt(la.det(S[2:, 2:]))
    bx = S[0, 0] / ex
    by = S[2, 2] / ey
    ax = -S[0, 1] / ex
    ay = -S[2, 3] / ey
    e1, e2 = get_mode_emittances(S)
    return ax, ay, bx, by, ex, ey, e1, e2

    
def read_stats(filename, drop_s):
    """Read turn-by-turn statistics files.
    
    Inputs
    ------
    filename : str
        The name of the file to be read containing the transverse beam moments.
        The columns are ['s','x2','xxp','xy','xyp','xp2','yxp','xpyp','y2',
        'yyp','yp2'], where s is the longitudinal position (always 0 for
        turn by turn data), x2 = <x^2>, xxp = <xx'>, etc.
        
    drop_s : bool
        If True, the 's' column is dropped.
        
    Returns
    ------
    Stats object
        Object with the following DataFrames as fields:
        * twiss : the Twiss parameters and emittances
        * moments : the 10 elements of the covariance matrix
        * corr : the 10 elements of the correlation matrix
        * beam : the tilt angle, radii and area of the real space ellipse
    """
    moments = pd.read_table(filename, sep=' ', names=['s'] + moment_cols)
    s_col = moments['s'].values
    if not drop_s:
        env_params = env_params.sort_values('s', ignore_index=True)
    moments = moments.loc[:, moments.columns != 's']
        
    nframes = moments.shape[0]
    corr = np.zeros((nframes, 10))
    twiss = np.zeros((nframes, 8))
    beam = np.zeros((nframes, 4))
    
    for i, moment_vec in enumerate(moments.values):
        cov_mat = vec2mat(moment_vec)
        corr_mat = cov2corr(cov_mat)
        corr[i] = mat2vec(corr_mat)
        twiss[i] = get_twiss(cov_mat)
        phi, cx, cy = rms_ellipse_params(moment_vec)
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
        def __init__(self, twiss, moments, corr, beam):
            self.twiss = twiss
            self.moments = moments
            self.corr = corr
            self.beam = beam
    
    return Stats(twiss, moments, corr, beam)

    
def read_coords(file_path, turns, mm_mrad=False):
    """Read turn-by-turn transverse coordinate data files.
    
    Parameters
    ----------
    file_path : str
        The path to data files. Ex: coords after turn 3 are found in
        'path/coords_3.dat'.
    turns : iterable
        The turns to read.
    
    Returns
    -------
    cdfs : list[DataFrame]
        List of coordinate DataFrames at each frame.
    """
    if file_path.endswith('/'):
        file_path = file_path[:-1]
        
    cdfs = []
    for turn in tqdm(turns):
        file = ''.join([file_path, '/coords_{}.dat'.format(turn)])
        cdf = pd.read_table(file, sep=' ', names=dims, usecols=list(range(4)))
        if mm_mrad:
            cdf *= 1e3
        cdfs.append(cdf)
    return cdfs


def normalize(cdfs, twiss):
    """Normalize the x-x' and y-y' projections.
            
    Parameters
    ----------
    cdfs : list[DataFrame]
        List of coordinate DataFrames at each frame.
    twiss : list
        The Twiss parameters to use: [ax, ay, bx, by, ex, ey].
        
    Returns
    -------
    cdfs : list[DataFrame]
        List of normalized coordinate DataFrames at each frame.
    """
    def norm(X, Vinv, ex, ey):
        Ainv = np.sqrt(np.diag([1/ex, 1/ex, 1/ey, 1/ey]))
        N = np.matmul(Ainv, Vinv)
        return np.apply_along_axis(lambda x: np.matmul(N, x), 1, X)
        
    ax, ay, bx, by, ex, ey = twiss
    V = norm_mat_4D(ax, bx, ay, by)
    Vinv = la.inv(V)
        
    cdfs_n = []
    for cdf in tqdm(cdfs):
        X = cdf.values
        X_n = norm(X, Vinv, ex, ey)
        cdf_n = pd.DataFrame(X_n, columns=['x','xp','y','yp'])
        cdfs_n.append(cdf_n)
    return cdfs_n
