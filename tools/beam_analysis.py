import numpy as np
import numpy.linalg as la
import pandas as pd

from .utils import mat2vec, vec2mat, cov2corr


env_cols = ['a','b','ap','bp','e','f','ep','fp']
moment_cols = ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2']
twiss2D_cols = ['ax','ay','bx','by','ex','ey']
twiss4D_cols = ['ax','ay','bx','by','u','nu','e1','e2','e4D']


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
    
    
def get_twiss2D(Sigma):
    """Return 2D Twiss parameters from covariance matrix."""
    eps_x, eps_y = apparent_emittances(Sigma)
    beta_x = Sigma[0, 0] / eps_x
    beta_y = Sigma[2, 2] / eps_y
    alpha_x = -Sigma[0, 1] / eps_x
    alpha_y = -Sigma[2, 3] / eps_y
    return np.array([alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y])
    
    
def get_twiss4D(Sigma, mode):
    """Return 4D Twiss parameters from covariance matrix. 
    
    This is technically only valid for the Danilov distribution. What we
    really need to do is compute V from the eigenvectors of Sigma U, then
    compute the Twiss parameters from V.
    """
    e1, e2 = intrinsic_emittances(Sigma)
    ex, ey = apparent_emittances(Sigma)
    eps = max([e1, e2])
    bx = Sigma[0, 0] / eps
    by = Sigma[2, 2] / eps
    ax = -Sigma[0, 1] / eps
    ay = -Sigma[2, 3] / eps
    nu = np.arccos(Sigma[0, 2] / np.sqrt(Sigma[0, 0]*Sigma[2, 2]))
    if mode == 1:
        u = ey / eps
    elif mode == 2:
        u = ex / eps
    return np.array([ax, ay, bx, by, u, nu, e1, e2, e1*e2])
    

class Stats:
    """Container for transverse beam statistics.
    
    Attributes
    ----------
    mode : {1, 2}
        The mode of the beam if it is supposed to describe a Danilov
        distribution. Currently, I don't know how to determine this from the
        covariance matrix. I can find the intrinsic emittances, but the order
        in which they are returned seems to be independent of the beam mode.
    twiss2D : pandas DataFrame
        2D Twiss parameters. The columns are:
            'ex': rms apparent emittance in x-x' plane
            'ey': rms apparent emittance in y-y' plane
            'bx': beta_x = <x^2> / ex
            'by': beta_y = <y^2> / ey
            'ax': alpha_x = -<xx'> / ex
            'ay': alpha_y = -<yy'> / ey
    twiss4D : pandas DataFrame
        4D Twiss parameters. In the following 'l' can be 1 or 2 depending on
        which of the two intrinsic emittances is nonzero. The columns are:
            'e1': rms intrinsic emittance for mode 1
            'e2': rms intrinsic emittance for mode 2
            'e4D': rms 4D emittance = e1 * e2
            'bx': beta_lx = <x^2> / el (l = 1 if e2=0, or 2 if e1=0)
            'by': beta_ly = <y^2> / el
            'ax': alpha_lx = -<xx'> / el
            'ay': alpha_ly = -<yy'> / el
            'u' : ey/el if l == 1, or ex/el if mode == 2
            'nu': the x-y phase difference in the beam. It is related to the
                  correlation coefficient as cos(nu) = x-y correlation
                  coefficent.
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
        Dimensions of the beam ellipse in real (x-y) space, where the ellipse
        is defined by 4 * x^T * Sigma * x, where Sigma is the covariance matrix
        and x = [x, x', y, y']. The columns are:
            'angle': the tilt angle (in degrees) measured below the x-axis.
            'cx' : the horizontal radius of the ellipse when `angle` is zero.
            'cy' : the vertical radius of the ellipse when `angle` is zero.
            'area': the area of the ellipse
            'area_rel' the area normalized by the initial area (the first row)
    env_params : pandas DataFrame
        The envelope parameters of the beam ellipse. If only the moments are
        provided, these are meaningless and are kept at zero.
    """
    def __init__(self, mode):
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
        self.nframes = data.shape[0]
        self.env_params_arr = np.zeros((self.nframes, 8))
        self.moments_arr = np.zeros((self.nframes, 10))
        self.corr_arr = np.zeros((self.nframes, 10))
        self.realspace_arr = np.zeros((self.nframes, 4))
        self.twiss2D_arr = np.zeros((self.nframes, 6))
        self.twiss4D_arr = np.zeros((self.nframes, 9))
        
    def read_moments(self, moments_list):
        if not self._initialized:
            self._create_empty_arrays(moments_list)
        for i, moments in enumerate(moments_list):
            cov_mat = vec2mat(moments)
            self.moments_arr[i] = moments
            self.corr_arr[i] = mat2vec(cov2corr(cov_mat))
            self.twiss2D_arr[i] = get_twiss2D(cov_mat)
            self.twiss4D_arr[i] = get_twiss4D(cov_mat, self.mode)
            angle, cx, cy = rms_ellipse_dims(cov_mat, 'x', 'y')
            cx *= 2 # Get real radii instead of rms
            cy *= 2 # Get real radii instead of rms
            angle = np.degrees(angle)
            self.realspace_arr[i] = [angle, cx, cy, np.pi*cx*cy]
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

    def _create_dfs(self):
        """Create pandas DataFrames from the ndarrays."""
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
        eps = self.twiss2D['ex'] + self.twiss2D['ey']
        self.twiss2D['ex_frac'] = self.twiss2D['ex'] / eps
        self.twiss2D['ey_frac'] = self.twiss2D['ey'] / eps
        
    def dfs(self):
        return [self.twiss2D, self.twiss4D, self.moments, self.corr,
                self.realspace, self.env_params]
