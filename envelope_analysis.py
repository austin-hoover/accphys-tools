"""
This module provides functions to analyze data describing the {2,2} Danilov
distribution.

The boundary of the beam ellipsoid in 4D phase space can be parameterized as:
    x = a * cos(psi) + b * sin(psi),
    x'= a'* cos(psi) + b'* sin(psi),
    x = e * cos(psi) + f * sin(psi),
    x'= e'* cos(psi) + f'* sin(psi),
with 0 <= psi <= 2pi. {a, b, a', b', e, f, e', f'} are called the 'envelope
parameters'.
"""

import numpy as np
from numpy import linalg as la
import pandas as pd
from scipy.integrate import trapz

from .utils import mat2vec, norm_mat_4D, cov2corr


env_cols = ['a','b','ap','bp','e','f','ep','fp']
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss_cols = ['ax','ay','bx','by','ex','ey','e1','e2']


def env_vec2mat(params):
    """Construct envelope matrix from envelope parameters."""
    a, b, ap, bp, e, f, ep, fp = params
    return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])


def get_coords(params, n_angles=50):
    """Get ellipse boundary data from envelope parameters.
    
    Parameters
    ----------
    params : array-like
        The envelope parameters [a, b, a', b', e, f, e', f'].
    n_angles : float
        Number of values of psi to calculate.
        
    Returns
    -------
    coords : NumPy array, shape (n_angles, 4)
        Columns are [x, x', y, y'], rows are different values of psi.
    """
    a, b, ap, bp, e, f, ep, fp = params
    psi = np.linspace(0, 2 * np.pi, n_angles)
    C, S = np.cos(psi), np.sin(psi)
    x  = a * C + b * S
    xp = ap * C + bp * S
    y  = e * C + f * S
    yp = ep * C + fp * S
    coords = np.vstack([x, xp, y, yp])
    return coords
    
    
def get_cov_mat(params):
    """Construct covariance matrix from envelope parameters."""
    S = np.zeros((4,4))
    a, b, ap, bp, e, f, ep, fp = params
    P = np.array([[a, b, 0, 0], [ap, bp, 0, 0], [e, f, 0, 0], [ep, fp, 0, 0]])
    return 0.25 * np.matmul(P, P.T)
    
    
def get_twiss(params):
    """Return Twiss parameters and emittances from sigma matrix."""
    S = get_cov_mat(params)
    ex = np.sqrt(la.det(S[:2, :2]))
    ey = np.sqrt(la.det(S[2:, 2:]))
    bx = S[0, 0] / ex
    by = S[2, 2] / ey
    ax = -S[0, 1] / ex
    ay = -S[2, 3] / ey
    e1, e2 = ex + ey, 0.0
    return ax, ay, bx, by, ex, ey, e1, e2


def tilt_angle(params):
    """Return tilt angle of ellipse in real space (ccw)."""
    a, b, ap, bp, e, f, ep, fp = params
    return 0.5 * np.arctan2(2*(a*e + b*f), a**2 + b**2 - e**2 - f**2)
    
    
def radii(params):
    """Return radii of ellipse in real space."""
    a, b, ap, bp, e, f, ep, fp = params
    phi = tilt_angle(params)
    cos, sin = np.cos(phi), np.sin(phi)
    cos2, sin2 = cos**2, sin**2
    a2b2, e2f2 = a**2 + b**2, e**2 + f**2
    area = a*f - b*e
    cx = np.sqrt(area**2 / (e2f2*cos2 + a2b2*sin2 +  2*(a*e + b*f)*cos*sin))
    cy = np.sqrt(area**2 / (a2b2*cos2 + e2f2*sin2 -  2*(a*e + b*f)*cos*sin))
    return cx, cy
    
    
def normalize(params, twiss_params):
    """Normalize the envelope parameters."""
    ax, ay, bx, by = twiss_params
    V = norm_mat_4D(ax, bx, ay, by)
    P = env_vec2mat(params)
    params_n = mat_to_vec(np.matmul(la.inv(V), P))
    return params_n


def parts_from_envelope(params_df, psi=0.0):
    """Compute single particle coordinates from envelope parameters."""
    C, S = np.cos(psi), np.sin(psi)
    part = pd.DataFrame()
    part['x'] = C * params_df['a'] + S * params_df['b']
    part['y'] = C * params_df['e'] + S * params_df['f']
    part['xp'] = C * params_df['ap'] + S * params_df['bp']
    part['yp'] = C * params_df['ep'] + S * params_df['fp']
    return part


def read(filename, drop_s=False, mm_mrad=True):
    """Read envelope parameters and calculate statistics.
    
    Parameters
    ----------
    filename : str
        Columns of file should be [s, a, b, a', b', e, f, e', f'], where s is
        the longitudinal position.
    drop_s : bool
        If True, the 's' column is dropped.
    mm_mrad : bool
        If True, convert the m-rad data to mm-mrad.
    
    Returns
    -------
    env_params : Pandas DataFrame
        The envelope parameters as at each frame.
    Stats object
        Object with the following DataFrames as fields:
        * twiss : the Twiss parameters and emittances
        * moments : the 10 elements of the covariance matrix
        * corr : the 10 elements of the correlation matrix
        * beam : the tilt angle, radii and area of the real space ellipse
    """
    env_params = pd.read_table(filename, sep=' ', names=['s'] + env_cols)
    if not drop_s:
        env_params = env_params.sort_values('s', ignore_index=True)
    s_col = env_params['s'].values
    env_params = env_params.loc[:, env_params.columns != 's']
    if mm_mrad:
        env_params *= 1000

    nframes = env_params.shape[0]
    moments = np.zeros((nframes, 10))
    corr = np.zeros((nframes, 10))
    twiss = np.zeros((nframes, 8))
    beam = np.zeros((nframes, 4))

    for i, params in enumerate(env_params.values):
        cov_mat = get_cov_mat(params)
        corr_mat = cov2corr(cov_mat)
        moments[i] = mat2vec(cov_mat)
        corr[i] = mat2vec(corr_mat)
        twiss[i] = get_twiss(params)
        phi = np.degrees(tilt_angle(params))
        cx, cy = radii(params)
        area = np.pi * cx * cy
        beam[i] = [phi, cx, cy, area]

    # Create DataFrames
    moments = pd.DataFrame(moments, columns=moment_cols)
    corr = pd.DataFrame(corr, columns=moment_cols)
    twiss = pd.DataFrame(twiss, columns=twiss_cols)
    beam = pd.DataFrame(beam, columns=['phi','cx','cy','area'])
    moments[['x_rms','y_rms']] = np.sqrt(moments[['x2','y2']])
    moments[['xp_rms','yp_rms']] = np.sqrt(moments[['xp2','yp2']])
    beam['area_rel'] = beam['area'] / beam.loc[0, 'area']
    
    if not drop_s:
        moments['s'] = corr['s'] = beam['s'] = twiss['s'] = env_params['s'] = s_col
        
    class Stats:
        def __init__(self, twiss, moments, corr, beam):
            self.twiss = twiss
            self.moments = moments
            self.corr = corr
            self.beam = beam
    
    return env_params, Stats(twiss, moments, corr, beam)
