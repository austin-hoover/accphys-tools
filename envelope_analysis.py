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

# Python
import sys

# 3rd party
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import pandas as pd
from scipy.integrate import trapz

# PyORBIT
sys.path.append('/Users/46h/Research/code/py-orbit/py/orbit')
from coupling import bogacz_lebedev as BL

# My modules
from .utils import (
    mat2vec,
    cov2corr,
    phase_adv_matrix,
    norm_mat
)

# Module level variables
env_cols = ['a','b','ap','bp','e','f','ep','fp']
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss_cols_2D = ['ax','ay','bx','by','ex','ey']
twiss_cols_4D = ['ax','ay','bx','by','u','nu']


def get_coords(params, nparts=50):
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
    psi = np.linspace(0, 2 * np.pi, nparts)
    cos, sin = np.cos(psi), np.sin(psi)
    x, xp = a*cos + b*sin, ap*cos + bp*sin
    y, yp = e*cos + f*sin, ep*cos + fp*sin
    coords = np.vstack([x, xp, y, yp]).T
    return coords
    
    
def get_ellipse_coords(params_list, npts=50):
    """Get ellipse coordinates at each frame.
    
    env_params_list : array like, shape (nframes, 8)
    """
    return np.array([get_coords(params, npts) for params in params_list])
    
    
def get_coord_array(params_list, nparts):
    """Generate ideal particle trajectories on the envelope.
    
    Returns NumPy array with shape (nframes, nparts, 4).
    """
    return np.array([get_coords(params, nparts) for params in params_list])

def read(filename, positions=None):
    """Read envelope parameters and calculate statistics.
    
    Parameters
    ----------
    filename : str
        Columns of file should be [a, b, a', b', e, f, e', f'].
    positions : array-like
        List of position
    
    Returns
    -------
    env_params : Pandas DataFrame
        A DataFrame containing the envelope parameters at each frame.
    Stats object
        Object with the following DataFrames as fields:
        * twiss : the Twiss parameters and emittances
        * moments : the 10 elements of the covariance matrix
        * corr : the 10 elements of the correlation matrix
        * beam : the tilt angle, radii and area of the real space ellipse
    """
    env_params = pd.read_table(filename, sep=' ', names=env_cols)

    nframes = env_params.shape[0]
    moments = np.zeros((nframes, 10))
    corr = np.zeros((nframes, 10))
    twiss2D = np.zeros((nframes, 6))
    twiss4D = np.zeros((nframes, 6))
    beam = np.zeros((nframes, 4))

    for i, params in enumerate(env_params.values):
        env = Envelope(params=params)
        moments[i] = mat2vec(env.cov())
        corr[i] = mat2vec(env.corr())
        twiss2D[i], twiss4D[i] = env.twiss(), env.twissBL()
        phi, (cx, cy) = env.tilt_angle(), env.radii()
        beam[i] = [np.degrees(phi), cx, cy, np.pi*cx*cy]

    # Create DataFrames
    moments = pd.DataFrame(moments, columns=moment_cols)
    corr = pd.DataFrame(corr, columns=moment_cols)
    twiss2D = pd.DataFrame(twiss2D, columns=twiss_cols_2D)
    twiss4D = pd.DataFrame(twiss4D, columns=twiss_cols_4D)
    
    # Add/edit columns
    beam = pd.DataFrame(beam, columns=['phi','cx','cy','area'])
    moments[['x_rms','y_rms']] = np.sqrt(moments[['x2','y2']])
    moments[['xp_rms','yp_rms']] = np.sqrt(moments[['xp2','yp2']])
    beam['area_rel'] = beam['area'] / beam.loc[0, 'area']
    twiss4D['nu'] = np.degrees(twiss4D['nu'])
        
    class Stats:
        """Container for beam statistics. Each attribute is a DataFrame."""
        def __init__(self, twiss2D, twiss4D, moments, corr, beam):
            self.twiss2D = twiss2D
            self.twiss4D = twiss4D
            self.moments = moments
            self.corr = corr
            self.beam = beam
            self.dfs = [self.twiss2D, self.twiss4D, self.moments,
                        self.corr,self.beam]
    
    return env_params, Stats(twiss2D, twiss4D, moments, corr, beam)


class Envelope:
    """Class for the Danilov distribution envelope.

    Attributes
    ----------
    params : NumPy array, shape (8,)
        The envelope parameters [a, b, a', b', e, f, e', f']. The coordinates
        of a particle on the beam envelope are parameterized as
            x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
            y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi),
        where 0 <= psi <= 2pi.
    eps : float
        The rms mode emittance of the beam [m*rad].
    mode : int
        Whether to choose eps2=0 (mode 1) or eps1=0 (mode 2).
    """

    def __init__(self, eps=1., mode=1, params=None, u=0.5):
        self.eps = eps
        self.mode = mode
        if params is not None:
            self.params = np.array(params)
            _, _, _, _, ex, ey = self.twiss()
            self.eps = ex + ey
        else:
            ex, ey = u * eps, (1 - u) * eps
            rx, ry = np.sqrt(4 * ex), np.sqrt(4 * ey)
            if mode == 1:
                self.params = np.array([rx, 0, 0, rx, 0, -ry, +ry, 0])
            elif mode == 2:
                self.params = np.array([rx, 0, 0, rx, 0, +ry, -ry, 0])
                
    def norm(self):
        """Return envelope to normalized frame.
        
        In this frame the covariance matrix is diagonal, and the x-x' and y-y'
        emittances are the mode emittances.
        """
        r_n = np.sqrt(4 * self.eps)
        if self.mode == 1:
            self.params = np.array([r_n, 0, 0, r_n, 0, 0, 0, 0])
        elif self.mode == 2:
            self.params = np.array([0, 0, 0, 0, 0, r_n, r_n, 0])
                        
    def norm2D(self):
        """Normalize the envelope parameters in the 2D sense.
        
        Here 'normalized' means the x-x' and y-y' ellipses will be circles of
        radius 1.
        """
        if self.mode == 1:
            self.params = np.array([1, 0, 0, 1, 0, -1, +1, 0])
        elif self.mode == 2:
            self.params = np.array([1, 0, 0, 1, 0, +1, -1, 0])
            
    def normed2D(self):
        """Same as `norm2D` method, but does not modify the envelope."""
        P = self.matrix()
        ax, ay, bx, by, _, _ = self.twiss()
        V = norm_mat(ax, bx, ay, by)
        return self.to_vec(np.matmul(la.inv(V), P))
            
    def matrix(self):
        """Create the envelope matrix P from the envelope parameters.
        
        The matrix is defined by x = Pc, where x = [x, x', y, y']^T and
        c = [cos(psi), sin(psi)], with 0 <= psi <= 2pi. This is useful because
        any transformation to the particle coordinate vector x also done to P.
        For example, if x -> M.x, then P -> M.P.
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
        
    def to_vec(self, P):
        """Convert the envelope matrix to vector form."""
        return P.flatten()
                
    def transform(self, M):
        """Apply matrix M to the coordinates."""
        P_new = np.matmul(M, self.matrix())
        self.params = self.to_vec(P_new)
        
    def norm_transform(self, M):
        """Normalize, then apply M to the coordinates."""
        self.norm()
        self.transform(M)

    def swap_xy(self):
        """Exchange (x, x') <-> (y, y')."""
        self.params[:4], self.params[4:] = self.params[4:], self.params[:4]
        
    def cov(self):
        """Return the transverse covariance matrix."""
        P = self.matrix()
        return 0.25 * np.matmul(P, P.T)
        
    def corr(self):
        return cov2corr(self.cov())
        
    def twiss(self):
        """Return the horizontal/vertical Twiss parameters and emittances."""
        S = self.cov()
        ex = np.sqrt(la.det(S[:2, :2]))
        ey = np.sqrt(la.det(S[2:, 2:]))
        bx = S[0, 0] / ex
        by = S[2, 2] / ey
        ax = -S[0, 1] / ex
        ay = -S[2, 3] / ey
        return np.array([ax, ay, bx, by, ex, ey])
        
    def twissBL(self):
        """Return the mode Twiss parameters, as defined by Bogacz & Lebedev."""
        ax, ay, bx, by, ex, ey = self.twiss()
        if self.mode == 1:
            u = ey / self.eps
            bx *= (1 - u)
            ax *= (1 - u)
            by *= u
            ay *= u
        elif self.mode == 2:
            u = ex / self.eps
            bx *= u
            ax *= u
            by *= (1 - u)
            ay *= (1 - u)
        nu = self.phase_diff()
        return np.array([ax, ay, bx, by, u, nu])
        
    def tilt_angle(self, x1='x', x2='y'):
        """Return the ccw tilt angle in the x1-x2 plane."""
        a, b, ap, bp, e, f, ep, fp = self.params
        var_to_params = {
            'x': (a, b),
            'y': (e, f),
            'xp': (ap, bp),
            'yp': (ep, fp)
        }
        a, b = var_to_params[x1]
        e, f = var_to_params[x2]
        return 0.5 * np.arctan2(2*(a*e + b*f), a**2 + b**2 - e**2 - f**2)

    def radii(self, x1='x', x2='y'):
        """Return the semi-major and semi-minor axes in the x1-x2 plane."""
        a, b, ap, bp, e, f, ep, fp = self.params
        phi = self.tilt_angle(x1, x2)
        cos, sin = np.cos(phi), np.sin(phi)
        cos2, sin2 = cos**2, sin**2
        var_to_params = {
            'x': (a, b),
            'y': (e, f),
            'xp': (ap, bp),
            'yp': (ep, fp)
        }
        a, b = var_to_params[x1]
        e, f = var_to_params[x2]
        a2b2, e2f2 = a**2 + b**2, e**2 + f**2
        area = a*f - b*e
        cx = np.sqrt(
            area**2 / (e2f2*cos2 + a2b2*sin2 +  2*(a*e + b*f)*cos*sin))
        cy = np.sqrt(
            area**2 / (a2b2*cos2 + e2f2*sin2 -  2*(a*e + b*f)*cos*sin))
        return cx, cy
        
    def phases(self):
        """Return the horizontal/vertical phases (in range [0, 2pi] of a
         particle with x=a, x'=a', y=e, y'=e'."""
        a, b, ap, bp, e, f, ep, fp = self.normed2D()
        mux = -np.arctan2(ap, a)
        muy = -np.arctan2(ep, e)
        if mux < 0:
            mux += 2*np.pi
        if muy < 0:
            muy += 2*np.pi
        return mux, muy
        
    def phase_diff(self):
        """Return the x-y phase difference (nu) of all particles in the beam.
        The value returned is in the range [0, pi].
        """
        mux, muy = self.phases()
        nu = abs(muy - mux)
        return nu if nu < np.pi else 2*np.pi - nu
        
    def fit_cov(self, Sigma, verbose=0):
        """Fit the envelope to the covariance matrix Sigma."""
        def mismatch(params, Sigma):
            self.params = params
            return 1e12 * mat2vec(Sigma - self.cov())
        result = opt.least_squares(mismatch, self.params, args=(Sigma,),
                                   xtol=1e-12)
        return result.x
        
    def fit_twiss(self, ax, ay, bx, by, u=0.5):
        """Fit the envelope to the 2D Twiss parameters."""
        self.norm2D()
        ex, ey = u*self.eps, (1-u)*self.eps
        V = norm_mat(ax, bx, ay, by)
        A = np.sqrt(4 * np.diag([ex, ex, ey, ey]))
        self.transform(np.matmul(V, A))
        
    def fit_twissBL(self, twiss_params):
        """Fit the envelope to the BL Twiss parameters.
        
        `twiss_params` is an array containing the Bogacz-Lebedev Twiss
        parameters for a single mode: [ax, ay, bx, by, u, nu], where
            
        ax{y} : float
            The horizontal{vertical} alpha function.
        bx{y} : float
            The horizontal{vertical} beta function.
        u : float
            The coupling parameter in range [0, 1]. This is equal to ey/e1
            when mode=1 or ex/e2 when mode=2.
        nu : float
            The x-y phase difference in range [0, pi].
        """
        ax, ay, bx, by, u, nu = twiss_params
        V = BL.Vmat(ax, ay, bx, by, u, nu, self.mode)
        self.norm_transform(V)
        
    def get_part_coords(self, psi=0):
        """Return the coordinates of a single particle on the envelope."""
        a, b, ap, bp, e, f, ep, fp = self.params
        cos, sin = np.cos(psi), np.sin(psi)
        x = a*cos + b*sin
        y = e*cos + f*sin
        xp = ap*cos + bp*sin
        yp = ep*cos + fp*sin
        return np.array([x, xp, y, yp])
        
    def generate_dist(self, nparts, density='uniform'):
        """Generate a distribution of particles from the envelope.

        Parameters
        ----------
        nparts : int
            The number of particles in the bunch.

        Returns
        -------
        X : NumPy array, shape (nparts, 4)
            The coordinate array for the distribution.
        """
        nparts = int(nparts)
        psis = np.linspace(0, 2*np.pi, nparts)
        X = np.array([self.get_part_coords(psi) for psi in psis])
        if density == 'uniform':
            radii = np.sqrt(np.random.random(nparts))
        elif density == 'on_ellipse':
            radii = np.ones(nparts)
        elif density == 'gaussian':
            radii = np.random.normal(size=nparts)
        return radii[:, np.newaxis] * X
