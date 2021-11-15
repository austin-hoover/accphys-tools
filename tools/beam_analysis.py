import numpy as np
import numpy.linalg as la
import pandas as pd

from .utils import cov2corr, symmetrize


env_cols = ['a','b','ap','bp','e','f','ep','fp']
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss2D_cols = ['alpha_x','alpha_y','beta_x','beta_y','eps_x','eps_y']
twiss4D_cols = ['alpha_x','alpha_y','beta_x','beta_y','u','nu','eps_1','eps_2','eps_4D','eps_4D_app','C']


def mat2vec(Sigma):
    """Return vector of independent elements in 4x4 symmetric matrix Sigma."""
    return Sigma[np.triu_indices(4)]
                  
                  
def vec2mat(moment_vec):
    """Inverse of `mat2vec`."""
    Sigma = np.zeros((4, 4))
    indices = np.triu_indices(4)
    for moment, (i, j) in zip(moment_vec, zip(*indices)):
        Sigma[i, j] = moment
    return symmetrize(Sigma)


def get_ellipse_coords(env_params, npts=100):
    """Get (x, y) coordinates along ellipse boundary from envelope parameters.
    
    This function is specific to the Danilov distribution.

    Parameters
    ----------
    params : array-like
        The envelope parameters [a, b, a', b', e, f, e', f'].
    npts : float
        Number of points along the ellipse.
        
    Returns
    -------
    coords : ndarray, shape (npts, 4)
        Columns are [x, x', y, y'].
    """
    a, b, ap, bp, e, f, ep, fp = env_params
    psi = np.linspace(0, 2 * np.pi, npts)
    cos, sin = np.cos(psi), np.sin(psi)
    x, xp = a*cos + b*sin, ap*cos + bp*sin
    y, yp = e*cos + f*sin, ep*cos + fp*sin
    coords = np.vstack([x, xp, y, yp]).T
    return coords


def _rms_ellipse_dims(sig_xx, sig_yy, sig_xy):
    """Return semi-axes and tilt angle of the RMS ellipse in the x-y plane."""
    angle = -0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)
    sn, cs = np.sin(angle), np.cos(angle)
    cx = np.sqrt(abs(sig_xx*cs**2 + sig_yy*sn**2 - 2*sig_xy*sn*cs))
    cy = np.sqrt(abs(sig_xx*sn**2 + sig_yy*cs**2 + 2*sig_xy*sn*cs))
    return angle, cx, cy

    
def rms_ellipse_dims(Sigma, x1='x', x2='y'):
    """Return (angle, c1, c2) of rms ellipse in x1-x2 plane, where angle is the
    clockwise tilt angle and c1/c2 are the semi-axes.
    """
    str_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}
    i, j = str_to_int[x1], str_to_int[x2]
    sii, sjj, sij = Sigma[i, i], Sigma[j, j], Sigma[i, j]
    angle = -0.5 * np.arctan2(2*sij, sii-sjj)
    sin, cos = np.sin(angle), np.cos(angle)
    sin2, cos2 = sin**2, cos**2
    c1 = np.sqrt(abs(sii*cos2 + sjj*sin2 - 2*sij*sin*cos))
    c2 = np.sqrt(abs(sii*sin2 + sjj*cos2 + 2*sij*sin*cos))
    return angle, c1, c2
    
    
def intrinsic_emittances(Sigma):
    """Return intrinsic emittances from covariance matrix."""
    U = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    trSU2 = np.trace(la.matrix_power(np.matmul(Sigma, U), 2))
    detS = la.det(Sigma)
    eps_1 = 0.5 * np.sqrt(-trSU2 + np.sqrt(trSU2**2 - 16 * detS))
    eps_2 = 0.5 * np.sqrt(-trSU2 - np.sqrt(trSU2**2 - 16 * detS))
    return eps_1, eps_2
    
    
def apparent_emittances(Sigma):
    """Return apparent emittances from covariance matrix."""
    eps_x = np.sqrt(la.det(Sigma[:2, :2]))
    eps_y = np.sqrt(la.det(Sigma[2:, 2:]))
    return eps_x, eps_y


def emittances(Sigma):
    """Return apparent and intrinsic emittances from covariance matrix."""
    eps_x, eps_y = apparent_emittances(Sigma)
    eps_1, eps_2 = intrinsic_emittances(Sigma)
    return eps_x, eps_y, eps_1, eps_2


def coupling_coefficient(Sigma):
    """This is not the standard definition."""
    eps_1, eps_2 = intrinsic_emittances(Sigma)
    eps_x, eps_y = apparent_emittances(Sigma)
    return 1.0 - np.sqrt((eps_1 * eps_2) / (eps_x * eps_y))
    
    
def twiss2D(Sigma):
    """Return 2D Twiss parameters from covariance matrix."""
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma[0, 0] / eps_x
    beta_y = Sigma[2, 2] / eps_y
    alpha_x = -Sigma[0, 1] / eps_x
    alpha_y = -Sigma[2, 3] / eps_y
    return np.array([alpha_x, alpha_y, beta_x, beta_y])
    
    
def twiss4D(Sigma, mode):
    """Return 4D Twiss parameters from covariance matrix. 
    
    This is technically only valid for the Danilov distribution. What we
    really need to do is compute V from the eigenvectors of Sigma U, then
    compute the Twiss parameters from V.
    """
    eps_1, eps_2 = intrinsic_emittances(Sigma)
    eps_x, eps_y = apparent_emittances(Sigma)
    eps_l = max([eps_1, eps_2])
    beta_lx = Sigma[0, 0] / eps_l
    beta_ly = Sigma[2, 2] / eps_l
    alpha_lx = -Sigma[0, 1] / eps_l
    alpha_ly = -Sigma[2, 3] / eps_l
    nu = np.arccos(Sigma[0, 2] / np.sqrt(Sigma[0, 0]*Sigma[2, 2]))
    if mode == 1:
        u = eps_y / eps_l
    elif mode == 2:
        u = eps_x / eps_l
    return np.array([alpha_lx, alpha_ly, beta_lx, beta_ly, u, nu])



class BeamStats:
    """Container for transverse beam statistics.
    
    Attributes
    ----------
    mode : {1, 2}
        The mode of the beam if it is supposed to describe a Danilov
        distribution. Currently, I don't know how to determine this from the
        covariance matrix. I can find the intrinsic emittances, but the order
        in which theps_y are returned seems to be independent of the beam mode.
    twiss2D : pandas DataFrame
        2D Twiss parameters. The columns are:
            'eps_x': rms apparent emittance in x-x' plane
            'eps_y': rms apparent emittance in y-y' plane
            'beta_x': beta_x = <x^2> / eps_x
            'beta_y': beta_y = <y^2> / eps_y
            'alpha_x': alpha_x = -<xx'> / eps_x
            'alpha_y': alpha_y = -<yy'> / eps_y
    twiss4D : pandas DataFrame
        4D Twiss parameters. In the following 'l' can be 1 or 2 depending on
        which of the two intrinsic emittances is nonzero. The columns are:
            'eps_1': rms intrinsic emittance for mode 1
            'eps_2': rms intrinsic emittance for mode 2
            'eps_4D': rms 4D emittance = eps_1 * eps_2
            'eps_4D_app': apparent rms 4D emittance = eps_x * eps_y
            'beta_lx': <x^2> / eps_l (l = 1 if eps_2 = 0, or 2 if eps_1 = 0)
            'beta_ly': <y^2> / eps_l
            'alpha_lx': -<xx'> / eps_l
            'alpha_ly': -<yy'> / eps_l
            'u' : eps_y / eps_l if l == 1, or eps_x / eps_l if mode = 2
            'nu': the x-y phase difference in the beam. It is related to the
                  correlation coefficient as cos(nu) = x-y correlation
                  coefficent.
            'C': coupling coefficient sqrt((eps_x * eps_y) / (eps_1 * eps_2))
        Technically, the above definitions are only true if the 4D emittance is
        zero. So maybe we should compute directly from the definition in
        Lebedev/Bogacz (I think I had trouble with this before).
    moments : pandas DataFrame
        The 10 transverse beam moments. Columns are labeled 'x2' for <x^2>,
        'xxp' for <xx'>, etc.
    corr : pandas DataFrame
        The 10 transverse beam correlation coefficents. Columns labels are the
        same as `moments`.
    realspace : pandas DataFrame
        Dimensions of the rms beam ellipse in real (x-y) space. The ellipse is 
        defined by x^T * Sigma * x = 1, where Sigma is the covariance matrix
        and x = [x, x', y, y']. The columns are:
            'angle': the tilt angle (in degrees) measured below the x-alpha_xis.
            'cx' : the horizontal radius of the ellipse when `angle` is zero.
            'cy' : the vertical radius of the ellipse when `angle` is zero.
            'area': the area of the ellipse
            'area_rel' the area normalized beta_y the initial area (the first row)
    env_params : pandas DataFrame
        The envelope parameters of the beam ellipse. If only the moments are
        provided, these are meaningless and are kept at zero.
    """
    def __init__(self, mode=None):
        if mode is None:
            mode = 1
        self.mode = mode
        self.twiss2D = None
        self.twiss4D = None
        self.moments = None
        self.corr = None
        self.realspace = None
        self.env_params = None
        self._initialized = False
        
    def _create_empty_arrays(self, data):
        self._initialized = True
        self.nframes = len(data)
        self.env_params_arr = np.zeros((self.nframes, 8))
        self.moments_arr = np.zeros((self.nframes, 10))
        self.corr_arr = np.zeros((self.nframes, 10))
        self.realspace_arr = np.zeros((self.nframes, 4))
        self.twiss2D_arr = np.zeros((self.nframes, 6))
        self.twiss4D_arr = np.zeros((self.nframes, 11))
        
    def read_cov(self, Sigma_list):
        moments_list = [mat2vec(Sigma) for Sigma in Sigma_list]
        return self.read_moments(moments_list)
        
    def read_moments(self, moments_list):
        if not self._initialized:
            self._create_empty_arrays(moments_list)
        for i, moments in enumerate(moments_list):
            Sigma = vec2mat(moments)
            Corr = cov2corr(Sigma)
            alpha_x, alpha_y, beta_x, beta_y = twiss2D(Sigma)
            alpha_lx, alpha_ly, beta_lx, beta_ly, u, nu = twiss4D(Sigma, self.mode)
            eps_x, eps_y = apparent_emittances(Sigma)
            eps_1, eps_2 = intrinsic_emittances(Sigma)
            eps_4D = eps_1 * eps_2
            eps_4D_app = eps_x * eps_y
            C = np.sqrt(eps_4D_app / eps_4D) if eps_4D > 0. else np.inf
            self.moments_arr[i] = moments
            self.corr_arr[i] = mat2vec(Corr)
            self.twiss2D_arr[i] = [alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y]
            self.twiss4D_arr[i] = [alpha_lx, alpha_ly, beta_lx, beta_ly, 
                                   u, nu, eps_1, eps_2, eps_4D, eps_4D_app, C]
            angle, cx, cy = rms_ellipse_dims(Sigma, 'x', 'y')
            angle = np.degrees(angle)
            area = np.pi * cx * cy
            self.realspace_arr[i] = [angle, cx, cy, area]
        self._create_dfs()
        
    def read_env(self, env_params_list):
        if not self._initialized:
            self._create_empty_arrays(env_params_list)
        moments_list = []
        for env_params in env_params_list:
            a, b, ap, bp, e, f, ep, fp = env_params
            P = np.array([[a, b], [ap, bp], [e, f], [ep, fp]])
            Sigma = 0.25 * np.matmul(P, P.T)
            moments_list.append(mat2vec(Sigma))
        return self.read_moments(moments_list)
    
    def read_coords(self, coords):
        moments_list = []
        for X in coords:
            X = X[:4, :4]
            Sigma = np.cov(X.T)
            moments = mat2vec(Sigma)
            moments_list.append(moments)
        self.read_moments(moments_list)

    def _create_dfs(self):
        self.env_params = pd.DataFrame(self.env_params_arr, columns=env_cols)
        self.moments = pd.DataFrame(self.moments_arr, columns=moment_cols)
        self.corr = pd.DataFrame(self.corr_arr, columns=moment_cols)
        self.twiss2D = pd.DataFrame(self.twiss2D_arr, columns=twiss2D_cols)
        self.twiss4D = pd.DataFrame(self.twiss4D_arr, columns=twiss4D_cols)
        self.realspace = pd.DataFrame(self.realspace_arr,
                                      columns=['angle','cx','cy','area'])
        # Add/edit columns
        self.moments[['x_rms','y_rms']] = np.sqrt(self.moments[['x2','y2']])
        self.moments[['xp_rms','yp_rms']] = np.sqrt(self.moments[['xp2','yp2']])
        self.realspace['area_rel'] = self.realspace['area'] / \
                                     self.realspace.loc[0, 'area']
        self.twiss4D['nu'] = np.degrees(self.twiss4D['nu'])
        eps = self.twiss2D['eps_x'] + self.twiss2D['eps_y']
        self.twiss2D['eps_x_frac'] = self.twiss2D['eps_x'] / eps
        self.twiss2D['eps_y_frac'] = self.twiss2D['eps_y'] / eps
        
    def dfs(self):
        return [self.twiss2D, self.twiss4D, self.moments, 
                self.corr, self.realspace, self.env_params]

    
class TuneCalculator:
    """This is a workaround while the PyORBIT method is broken.
    
    Note: we normalize by the lattice Twiss parameters, not the beam Twiss parameters. 
    """
    def __init__(self, mass, kin_energy, alpha_x, alpha_y, beta_x, beta_y, eta_x=0., eta_px=0.):
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.eta_x = eta_x
        self.eta_px = eta_px
        self.kin_energy = kin_energy
        self.mass = mass
        self.energy = self.kin_energy + self.mass
        self.gamma = self.energy / self.mass
        self.beta = 1 / np.sqrt(1 - (1 / self.gamma**2))
        
    def normalize(self, X):
        Xn = np.copy(X)
        for i, (x, xp, y, yp, z, dE) in enumerate(X):
            dpp = (1 / self.beta**2) * dE / self.energy
            xn = (x - self.eta_x * dpp) / np.sqrt(self.beta_x)
            xpn = (xp - self.eta_px * dpp) * np.sqrt(self.beta_x) + xn * self.alpha_x
            yn = y / np.sqrt(self.beta_y)
            ypn = (yp + y * self.alpha_y / self.beta_y) * np.sqrt(self.beta_y)
            Xn[i, :] = [xn, xpn, yn, ypn, z, dE]
        return Xn

    def get_phases(self, X): 
        def _get_phases(u, up):
            phases = np.arctan2(up, u)
            phases[np.where(phases < 0.)] += (2 * np.pi)
            return phases
        X = self.normalize(X)
        xphases = _get_phases(X[:, 0], X[:, 1])
        yphases = _get_phases(X[:, 2], X[:, 3])
        return np.vstack([xphases, yphases]).T
    
    def get_tunes(self, X0, X1):
        phases0 = self.get_phases(X0)
        phases1 = self.get_phases(X1)
        # The bunches could be different sizes.
        min_n_parts = min(len(phases0), len(phases1))
        phases0 = phases0[:min_n_parts, :]
        phases1 = phases1[:min_n_parts, :]
        tunes = (phases0 - phases1) / (2 * np.pi)
        tunes[np.where(tunes < 0.)] += 1.
        return tunes
    
    
def load_pybunch(filename):
    X = pd.read_table(filename, sep=' ', skiprows=15, index_col=False, 
                      names=['x','xp','y','yp','z','dE', 'mux', 'muy', 'nux', 'nuy', 'Jx', 'Jy'])
    X.iloc[:, :4] *= 1000. # convert from m-rad to mm-mrad
    X.iloc[:, 5] *= 1000. # convert energy spread from [GeV] to [MeV]
    return X