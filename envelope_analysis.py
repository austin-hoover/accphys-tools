"""
This module contains functions related to the envelope parameterization of
the {2, 2} Danilov distribution.

Reference: https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.6.094202
"""

# Standard
import sys
# Third party
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.optimize as opt
from scipy.integrate import trapz
# Local
from .utils import (
    mat2vec,
    cov2corr,
    phase_adv_matrix,
    Vmat_2D
)

# Module level variables
env_cols = ['a','b','ap','bp','e','f','ep','fp']
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss_cols_2D = ['ax','ay','bx','by','ex','ey']
twiss_cols_4D = ['ax','ay','bx','by','u','nu']


# Functions
#------------------------------------------------------------------------------
def tprint(string, indent=4):
    """Print with indent."""
    print(indent*' ' + str(string))


def rotation_matrix(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])
    
    
def rotation_matrix_4D(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, 0, S, 0], [0, C, 0, S], [-S, 0, C, 0], [0, -S, 0, C]])

    
def phase_adv_matrix(mu1, mu2):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(mu1)
    R[2:, 2:] = rotation_matrix(mu2)
    return R


def Vmat_4D(ax, ay, bx, by, u, nu, mode=1):
    """Construct V from the Twiss parameters for the case when one of the
    intrinsic emittances is zero."""
    cos, sin = np.cos(nu), np.sin(nu)
    V = np.zeros((4, 4))
    if mode == 1:
        V[:2, :2] = [[np.sqrt(bx), 0],
                     [-ax/np.sqrt(bx), (1-u)/np.sqrt(bx)]]
        V[2:, :2] = [[np.sqrt(by)*cos, -np.sqrt(by)*sin],
                     [(u*sin-ay*cos)/np.sqrt(by), (u*cos + ay*sin)/np.sqrt(by)]]
    elif mode == 2:
        V[2:, 2:] = [[np.sqrt(by), 0],
                     [-ay/np.sqrt(by), (1-u)/np.sqrt(by)]]
        V[:2, 2:] = [[np.sqrt(bx)*cos, -np.sqrt(bx)*sin],
                     [(u*sin-ax*cos)/np.sqrt(bx), (u*cos + ax*sin)/np.sqrt(bx)]]
    return V


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

def compute_stats(params):
    """Read envelope parameters and calculate statistics.
    
    Parameters
    ----------
    params : NumPy array, shape (nframes, 8)
        The envelope parameters at each frame. Columns are [a, b, a', b', e,
        f, e', f'].
    
    Returns
    -------
    env_params : Pandas DataFrame, shape (nframes, 8)
        A DataFrame containing the envelope parameters at each frame.
    Stats object
        Object with the following DataFrames as fields:
        * moments : the 10 elements of the covariance matrix
        * corr : the 10 elements of the correlation matrix
        * beam : the tilt angle, radii and area of the real space ellipse
        * twiss2D : the 2D Twiss parameters and emittances
        * twiss4D : the 4D Twiss parameters and emittances
    """
    params_df = pd.DataFrame(params, columns=env_cols)

    nframes = params.shape[0]
    moments = np.zeros((nframes, 10))
    corr = np.zeros((nframes, 10))
    beam = np.zeros((nframes, 4))
    twiss2D = np.zeros((nframes, 6))
    twiss4D = np.zeros((nframes, 6))

    for i, params in enumerate(params):
        env = Envelope(params=params)
        moments[i] = mat2vec(env.cov())
        corr[i] = mat2vec(env.corr())
        twiss2D[i, :4] = env.twiss2D()
        twiss2D[i, 4:] = env.emittances()
        twiss4D[i] = env.twiss4D()
        phi = env.tilt_angle()
        cx, cy = env.radii()
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
    mode_emittance = twiss2D['ex'] + twiss2D['ey']
    twiss2D['ex_frac'] = twiss2D['ex'] / mode_emittance
    twiss2D['ey_frac'] = twiss2D['ey'] / mode_emittance
        
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
    
    return params_df, Stats(twiss2D, twiss4D, moments, corr, beam)


# Class definitions
#------------------------------------------------------------------------------
class Envelope:
    """Class for the Danilov distribution envelope.

    Attributes
    ----------
    eps : float
        The rms intrinsic emittance of the beam [m*rad].
    mode : int
        Whether to choose eps2=0 (mode 1) or eps1=0 (mode 2).
    ex_frac : float
        The x emittance ratio, such that ex = ex_frac * eps
    params : list, optional
        The envelope parameters [a, b, a', b', e, f, e', f']. The coordinates
        of a particle on the beam envelope are parameterized as
            x = a*cos(psi) + b*sin(psi), x' = a'*cos(psi) + b'*sin(psi),
            y = e*cos(psi) + f*sin(psi), y' = e'*cos(psi) + f'*sin(psi),
        where 0 <= psi <= 2pi.
    """
    def __init__(self, eps=1., mode=1, ex_frac=0.5, mass=0.93827231,
                 energy=1., intensity=0., length=1e-5, params=None):
        self.eps = eps
        self.mode = mode
        self.ex_frac, self.ey_frac = ex_frac, 1 - ex_frac
        if params is not None:
            self.params = np.array(params)
            ex, ey = self.emittances()
            self.eps = ex + ey
            self.ex_frac = ex / self.eps
        else:
            ex, ey = ex_frac * eps, (1 - ex_frac) * eps
            rx, ry = np.sqrt(4 * ex), np.sqrt(4 * ey)
            if mode == 1:
                self.params = np.array([rx, 0, 0, rx, 0, -ry, +ry, 0])
            elif mode == 2:
                self.params = np.array([rx, 0, 0, rx, 0, +ry, -ry, 0])
        
    def set_params(self, a, b, ap, bp, e, f, ep, fp):
        self.params = np.array([a, b, ap, bp, e, f, ep, fp])
        
    def get_params_for_dim(self, dim='x'):
        """Return envelope parameters associated with the given dimension."""
        a, b, ap, bp, e, f, ep, fp = self.params
        return {'x':(a, b), 'y':(e, f), 'xp':(ap, bp), 'yp': (ep, fp)}[dim]
        
    def matrix(self):
        """Create the envelope matrix P from the envelope parameters.
        
        The matrix is defined by x = P.c, where x = [x, x', y, y']^T,
        c = [cos(psi), sin(psi)], and '.' means matrix multiplication, with
        0 <= psi <= 2pi. This is useful because any transformation to the
        particle coordinate vector x also done to P. For example, if x -> M.x,
        then P -> M.P.
        """
        a, b, ap, bp, e, f, ep, fp = self.params
        return np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
        
    def to_vec(self, P):
        """Convert the envelope matrix to vector form."""
        return P.ravel()
        
    def get_norm_mat_2D(self, inv=False):
        """Return the normalization matrix V (2D sense)."""
        ax, ay, bx, by = self.twiss2D()
        V = Vmat_2D(ax, bx, ay, by)
        return la.inv(V) if inv else V
        
    def norm4D(self):
        """Normalize the envelope parameters in the 4D sense.
        
        In the transformed coordates the covariance matrix is diagonal, and the
        x-x' and y-y' emittances are the intrinsic emittances.
        """
        r_n = np.sqrt(4 * self.eps)
        if self.mode == 1:
            self.params = np.array([r_n, 0, 0, r_n, 0, 0, 0, 0])
        elif self.mode == 2:
            self.params = np.array([0, 0, 0, 0, 0, r_n, r_n, 0])
                                
    def norm2D(self, scale=False):
        """Normalize the envelope parameters in the 2D sense and return the
        parameters.
        
        Here 'normalized' means the x-x' and y-y' ellipses will be circles of
        radius sqrt(ex) and sqrt(ey), where ex and ey are the apparent
        emittances. The cross-plane elements of the covariance matrix will not
        all be zero. If `scale` is True, the x-x' and y-y' ellipses will be
        scaled to unit radius.
        """
        self.transform(self.get_norm_mat_2D(inv=True))
        if scale:
            ex, ey = 4 * self.emittances()
            self.params[:4] /= np.sqrt(ex)
            self.params[4:] /= np.sqrt(ey)
        return self.params
            
    def normed_params_2D(self):
        """Return the normalized envelope parameters in the 2D sense without
        actually changing the envelope."""
        true_params = np.copy(self.params)
        normed_params = self.norm2D()
        self.params = true_params
        return normed_params
                
    def transform(self, M):
        """Apply matrix M to the coordinates."""
        P_new = np.matmul(M, self.matrix())
        self.params = self.to_vec(P_new)
        
    def norm_transform(self, M):
        """Normalize, then apply M to the coordinates."""
        self.norm4D()
        self.transform(M)
        
    def advance_phase(self, mux=0., muy=0.):
        """Advance the x{y} phase by mux{muy} degrees.

        It is equivalent to tracking through an uncoupled lattice which the
        envelope is matched to.
        """
        mux, muy = np.radians([mux, muy])
        V = self.get_norm_mat_2D()
        M = la.multi_dot([V, phase_adv_matrix(mux, muy), la.inv(V)])
        self.transform(M)
        
    def rotate(self, phi):
        """Apply clockwise rotation by phi degrees in x-y space."""
        self.transform(rotation_matrix_4D(np.radians(phi)))

    def swap_xy(self):
        """Exchange (x, x') <-> (y, y')."""
        self.params[:4], self.params[4:] = self.params[4:], self.params[:4]
        
    def cov(self):
        """Return the transverse covariance matrix."""
        P = self.matrix()
        return 0.25 * np.matmul(P, P.T)
        
    def corr(self):
        """Return the transverse correlation matrix."""
        return cov2corr(self.cov())
        
    def emittances(self, mm_mrad=False):
        """Return the horizontal/vertical rms emittance."""
        Sigma = self.cov()
        ex = np.sqrt(la.det(Sigma[:2, :2]))
        ey = np.sqrt(la.det(Sigma[2:, 2:]))
        emittances = np.array([ex, ey])
        if mm_mrad:
            emittances *= 1e6
        return  emittances
        
    def twiss2D(self):
        """Return the horizontal/vertical Twiss parameters and emittances."""
        Sigma = self.cov()
        ex = np.sqrt(la.det(Sigma[:2, :2]))
        ey = np.sqrt(la.det(Sigma[2:, 2:]))
        bx = Sigma[0, 0] / ex
        by = Sigma[2, 2] / ey
        ax = -Sigma[0, 1] / ex
        ay = -Sigma[2, 3] / ey
        return np.array([ax, ay, bx, by])
        
    def twiss4D(self):
        """Return the 4D Twiss parameters, as defined by Bogacz & Lebedev."""
        Sigma = self.cov()
        ex = np.sqrt(la.det(Sigma[:2, :2]))
        ey = np.sqrt(la.det(Sigma[2:, 2:]))
        bx = Sigma[0, 0] / self.eps
        by = Sigma[2, 2] / self.eps
        ax = -Sigma[0, 1] / self.eps
        ay = -Sigma[2, 3] / self.eps
        nu = self.phase_diff()
        if self.mode == 1:
            u = ey / self.eps
        elif self.mode == 2:
            u = ex / self.eps
        return np.array([ax, ay, bx, by, u, nu])
        
    def tilt_angle(self, x1='x', x2='y'):
        """Return the ccw tilt angle in the x1-x2 plane."""
        a, b = self.get_params_for_dim(x1)
        e, f = self.get_params_for_dim(x2)
        return 0.5 * np.arctan2(2*(a*e + b*f), a**2 + b**2 - e**2 - f**2)

    def radii(self, x1='x', x2='y'):
        """Return the semi-major and semi-minor axes in the x1-x2 plane."""
        a, b = self.get_params_for_dim(x1)
        e, f = self.get_params_for_dim(x2)
        phi = self.tilt_angle(x1, x2)
        cos, sin = np.cos(phi), np.sin(phi)
        cos2, sin2, sincos = cos**2, sin**2, sin*cos
        x2, y2 = a**2 + b**2, e**2 + f**2
        A = abs(a*f - b*e)
        cx = np.sqrt(A**2 / (y2*cos2 + x2*sin2 + 2*(a*e + b*f)*sincos))
        cy = np.sqrt(A**2 / (x2*cos2 + y2*sin2 - 2*(a*e + b*f)*sincos))
        return np.array([cx, cy])
        
    def area(self, x1='x', x2='y'):
        """Return the area in the x1-x2 plane."""
        a, b = self.get_params_for_dim(x1)
        e, f = self.get_params_for_dim(x2)
        return np.pi * np.abs(a*f - b*e)
        
    def phases(self):
        """Return the horizontal/vertical phases in range [0, 2*pi] of a
         particle with x=a, x'=a', y=e, y'=e'."""
        a, b, ap, bp, e, f, ep, fp = self.normed_params_2D()
        mux, muy = -np.arctan2(ap, a), -np.arctan2(ep, e)
        if mux < 0:
            mux += 2*np.pi
        if muy < 0:
            muy += 2*np.pi
        return mux, muy
        
    def phase_diff(self):
        """Return the x-y phase difference (nu) of all particles in the beam.
        
        The value returned is in the range [0, pi]. This can also be found from
        the equation cos(nu) = r, where r is the x-y correlation coefficient.
        """
        mux, muy = self.phases()
        nu = abs(muy - mux)
        return nu if nu < np.pi else 2*np.pi - nu
        
    def fit_twiss2D(self, ax, ay, bx, by, ex_frac):
        """Fit the envelope to the 2D Twiss parameters."""
        V = Vmat_2D(ax, bx, ay, by)
        ex, ey = ex_frac * self.eps, (1 - ex_frac) * self.eps
        A = np.sqrt(4 * np.diag([ex, ex, ey, ey]))
        self.norm2D(scale=True)
        self.transform(np.matmul(V, A))
        
    def fit_twiss4D(self, twiss_params):
        """Fit the envelope to the 4D Twiss parameters.
        
        `twiss_params` is an array containing the 4D Twiss params for a single
        mode: [ax, ay, bx, by, u, nu], where
        * ax{y} : The horizontal{vertical} alpha function -<xx'>/e1 {-<xx'>/e2}.
        * bx{y} : The horizontal{vertical} beta function <xx>/e1 {<xx>/e2}.
        * u : The coupling parameter in range [0, 1]. This is equal to ey/e1
              when mode=1 or ex/e2 when mode=2.
        * nu : The x-y phase difference in range [0, pi].
        """
        ax, ay, bx, by, u, nu = twiss_params
        V = BL.Vmat(ax, ay, bx, by, u, nu, self.mode)
        self.norm_transform(V)
        
    def set_twiss_param_4D(self, name, value):
        """Change a single Twiss parameter while keeping the others fixed."""
        ax, ay, bx, by, u, nu = self.twiss4D()
        twiss_dict = {'ax':ax, 'ay':ay, 'bx':bx, 'by':by, 'u':u, 'nu':nu}
        twiss_dict[name] = value
        self.fit_twiss4D([twiss_dict[key]
                          for key in ('ax', 'ay', 'bx', 'by', 'u', 'nu')])
        
    def fit_cov(self, Sigma, verbose=0):
        """Fit the envelope to the covariance matrix Sigma."""
        def mismatch(params, Sigma):
            self.params = params
            return 1e12 * covmat2vec(Sigma - self.cov())
        result = opt.least_squares(mismatch, self.params, args=(Sigma,),
                                   xtol=1e-12, verbose=verbose)
        return result.x
        
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

        Returns: NumPy array, shape (nparts, 4)
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
        
    def match_bare(self, M, method='auto'):
        """Match to the lattice transfer matrix.
        
        Parameters
        ----------
        M : NumPy array, shape (4, 4)
            The lattice transfer matrix.
        method : str
            If '4D', match to the lattice using the eigenvectors of the
            transfer matrix. This may result in the beam being completely
            flat, for example when the lattice is uncoupled. The '2D' method
            will only match the x-x' and y-y' ellipses of the beam.
            
        Returns
        -------
        NumPy array, shape (8,)
            The matched envelope parameters.
        """
        if method == 'auto':
            method = '4D' if unequal_eigtunes(M) else '2D'
        if method == '2D':
            lattice_params = params_from_transfer_matrix(M)
            ax, ay = [lattice_params[key] for key in ('alpha_x', 'alpha_y')]
            bx, by = [lattice_params[key] for key in ('beta_x', 'beta_y')]
            self.fit_twiss2D(ax, ay, bx, by, self.ex_frac)
        elif method == '4D':
            eigvals, eigvecs = la.eig(M)
            V = BL.construct_V(eigvecs)
            self.norm_transform(V)
        return self.params
        
    def track(self, M, nturns):
        """Track the envelope using the transfer matrix `M`."""
        for _ in range(nturns):
            self.transform(M)

    def perturb(self, radius=0.1):
        """Randomly perturb the 4D Twiss parameters."""
        if radius == 0:
            return
        lo, hi = 1 - radius, 1 + radius
        ax, ay, bx, by, u, nu = self.twiss4D()
        ax_min, ax_max = lo*ax, hi*ax
        ay_min, ay_max = lo*ay, hi*ay
        bx_min, bx_max = lo*bx, hi*bx
        by_min, by_max = lo*by, hi*by
        u_min, u_max = lo*u, hi*u
        nu_min, nu_max = lo*nu, hi*nu
        if bx_min < 0.1:
            bx_min = 0.1
        if by_min < 0.1:
            by_min = 0.1
        if u_min < 0.05:
            u_min = 0.05
        if u_max > 0.95:
            u_max = 0.95
        if nu_min < 0.05 * np.pi:
            nu_min = 0.05 * np.pi
        if nu_max > 0.95 * np.pi:
            nu_max = 0.95 * np.pi
        ax = np.random.uniform(ax_min, ax_max)
        ay = np.random.uniform(ay_min, ay_max)
        bx = np.random.uniform(bx_min, bx_max)
        by = np.random.uniform(by_min, by_max)
        u = np.random.uniform(u_min, u_max)
        nu = np.random.uniform(nu_min, nu_max)
        twiss_params = (ax, ay, bx, by, u, nu)
        self.fit_twiss4D(twiss_params)
        
    def print_twiss2D(self, indent=4):
        (ax, ay, bx, by), (ex, ey) = self.twiss2D(), self.emittances()
        print('2D Twiss parameters:')
        tprint('ax, ay = {:.3f}, {:.3f} [rad]'.format(ax, ay))
        tprint('bx, by = {:.3f}, {:.3f} [m]'.format(bx, by))
        tprint('ex, ey = {:.3e}, {:.3e} [m*rad]'.format(ex, ey))

    def print_twiss4D(self):
        ax, ay, bx, by, u, nu = self.twiss4D()
        print('4D Twiss parameters:')
        tprint('mode = {}'.format(self.mode))
        tprint('e{} = {:.3e} [m*rad]'.format(self.mode, self.eps))
        tprint('ax, ay = {:.3f}, {:.3f} [rad]'.format(ax, ay))
        tprint('bx, by = {:.3f}, {:.3f} [m]'.format(bx, by))
        tprint('u = {:.3f}'.format(u))
        tprint('nu = {:.3f} [deg]'.format(u))
