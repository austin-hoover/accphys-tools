import numpy as np
import numpy.linalg as la


def rotation_matrix(phi):
    return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    
def phase_space_rotation_matrix(phi_x, phi_y):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(phi_x)
    R[2:, 2:] = rotation_matrix(phi_y)
    return R
    
def mat2vec(S):
    """Return vector of independent elements in 4x4 symmetric matrix S."""
    return np.array([S[0,0], S[0,1], S[0,2], S[0,3], S[1,1],
                     S[1,2], S[1,3], S[2,2], S[2,3], S[3,3]])
                     
def vec2mat(v):
    """Return 4x4 symmetric matrix from 10 element vector."""
    s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = v
    return np.array([[s11, s12, s13, s14],
                     [s12, s22, s23, s24],
                     [s13, s23, s33, s34],
                     [s14, s24, s34, s44]])
                     
def symmetrize(M):
    """Return a symmetrized version of M.
    
    M : NumPy array
        A square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())
    
def cov2corr(cov_mat):
    """Convert covariance matrix to correlation matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = la.inv(D)
    corr_mat = la.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat
    
def norm_mat_2D(alpha, beta):
    return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)

def norm_mat_4D(alpha_x, beta_x, alpha_y, beta_y):
    V = np.zeros((4, 4))
    V[:2, :2] = norm_mat_2D(alpha_x, beta_x)
    V[2:, 2:] = norm_mat_2D(alpha_y, beta_y)
    return V
    
def get_phase_adv(beta_x, beta_y, s, units='deg'):
    """Compute phase advance as function of s."""
    n_pts = len(s)
    phases = np.zeros((n_pts, 2))
    for i in range(n_pts):
        phases[i, 0] = trapz(1 / beta_x[:i], s[:i])
        phases[i, 1] = trapz(1 / beta_y[:i], s[:i]) # radians
    if units == 'deg':
        phases = np.degrees(phases)
    elif units == 'tune':
        phases /= 2*np.pi
    phases_df = pd.DataFrame(phases, columns=['x','y'])
    phases_df['s'] = s
    return phases_df
