import numpy as np
import numpy.linalg as la
from scipy.integrate import trapz


classical_proton_radius = 1.53469e-18 # [m]


def rotation_matrix(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


def rotation_matrix_4D(angle):
    """4x4 matrix to rotate [x, x', y, y'] clockwise in the x-y plane."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])


def phase_adv_matrix(mu1, mu2):
    """4x4 matrix to rotate x-x' by mu1 and y-y' by mu2, both clockwise."""
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(mu1)
    R[2:, 2:] = rotation_matrix(mu2)
    return R


def V_matrix_2x2(alpha, beta):
    """2x2 normalization matrix for x-x' or y-y'."""
    return np.array([[beta, 0], [-alpha, 1]]) / np.sqrt(beta)
    
    
def V_matrix_4x4_uncoupled(alpha_x, alpha_y, beta_x, beta_y):
    """4x4 normalization matrix for x-x' and y-y'."""
    V = np.zeros((4, 4))
    V[:2, :2] = V_matrix_2x2(alpha_x, beta_x)
    V[2:, 2:] = V_matrix_2x2(alpha_y, beta_y)
    return V


def Sigma_from_twiss2D(alpha_x, alpha_y, beta_x, beta_y, eps_x, eps_y):
    gamma_x = (1 + alpha_x**2) / beta_x
    gamma_y = (1 + alpha_y**2) / beta_y
    Sigma = np.zeros((4, 4))
    Sigma[:2, :2] = eps_x * np.array([[beta_x, -alpha_x], [-alpha_x, gamma_x]])
    Sigma[2:, 2:] = eps_y * np.array([[beta_y, -alpha_y], [-alpha_y, gamma_y]])
    return Sigma


def params_from_transfer_matrix(M):
    """Return dictionary of lattice parameters from a transfer matrix.
    
    Method is taken from `py/orbit/matrix_lattice/MATRIX_Lattice.py`.
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        A transfer matrix.
        
    Returns
    -------
    lattice_params : dict
        Dictionary with the following keys: 'frac_tune_x', 'frac_tune_y',
        'alpha_x', 'alpha_y', 'beta_x', 'beta_y', 'gamma_x', 'gamma_y'.
    """
    keys = ['frac_tune_x', 'frac_tune_y', 
            'alpha_x', 'alpha_y', 
            'beta_x', 'beta_y', 
            'gamma_x', 'gamma_y']
    lattice_params = {key: None for key in keys}
    
    cos_phi_x = (M[0, 0] + M[1, 1]) / 2
    cos_phi_y = (M[2, 2] + M[3, 3]) / 2
    if abs(cos_phi_x) >= 1 or abs(cos_phi_y) >= 1 :
        return lattice_params
    sign_x = sign_y = +1
    if abs(M[0, 1]) != 0:
        sign_x = M[0, 1] / abs(M[0, 1])
    if abs(M[2, 3]) != 0:
        sign_y = M[2, 3] / abs(M[2, 3])
    sin_phi_x = sign_x * np.sqrt(1 - cos_phi_x**2)
    sin_phi_y = sign_y * np.sqrt(1 - cos_phi_y**2)
    
    nux = sign_x * np.arccos(cos_phi_x) / (2 * np.pi)
    nuy = sign_y * np.arccos(cos_phi_y) / (2 * np.pi)
    beta_x = M[0, 1] / sin_phi_x
    beta_y = M[2, 3] / sin_phi_y
    alpha_x = (M[0, 0] - M[1, 1]) / (2 * sin_phi_x)
    alpha_y = (M[2, 2] - M[3, 3]) / (2 * sin_phi_y)
    gamma_x = -M[1, 0] / sin_phi_x
    gamma_y = -M[3, 2] / sin_phi_y
    
    lattice_params['frac_tune_x'] = nux
    lattice_params['frac_tune_y'] = nuy
    lattice_params['beta_x'] = beta_x
    lattice_params['beta_y'] = beta_y
    lattice_params['alpha_x'] = alpha_x
    lattice_params['alpha_y'] = alpha_y
    lattice_params['gamma_x'] = gamma_x
    lattice_params['gamma_y'] = gamma_y
    return lattice_params


def is_stable(M, tol=1e-5):
    """Return True if transfer matrix is stable.
    
    M : ndarray, shape (n, n)
        A transfer matrix.
    tol : float
        The matrix is stable if all eigenvalue norms are in the range 
        (1 - tol, 1 + tol).
    """
    for eigval in la.eigvals(M):
        if abs(la.norm(eigval) - 1.0) > tol:
            return False
    return True
    
    
def get_eigtunes(M):
    """Compute transfer matrix eigentunes -- cos(Re[eigenvalue]).
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        A transfer matrix.

    Returns
    -------
    ndarray, shape (2,)
        Eigentunes for the two modes.
    """
    return np.arccos(la.eigvals(M).real)[[0, 2]]
    

def unequal_eigtunes(M, tol=1e-5):
    """Return True if the eigentunes of the transfer matrix are the same.

    M : ndarray, shape (4, 4)
        A transfer matrix.
    tol : float
        Tunes are equal if abs(mu1 - mu2) > tol.
    """
    mu1, mu2 = get_eigtunes(M)
    return abs(mu1 - mu2) > tol
    
    
def mat2vec(Sigma):
    """Return vector of independent elements in 4x4 symmetric matrix Sigma."""
    return Sigma[np.triu_indices(4)]
                  
                  
def vec2mat(moment_vec):
    """Return 4x4 symmetric matrix from 10 element vector."""
    Sigma = np.zeros((4, 4))
    indices = np.triu_indices(4)
    for moment, (i, j) in zip(moment_vec, zip(*indices)):
        Sigma[i, j] = moment
    return symmetrize(Sigma)

    
def get_phase_adv(beta, positions, units='deg'):
    """Compute the phase advance by integrating the beta function."""
    npts = len(positions)
    phases = np.zeros(npts)
    for i in range(npts):
        phases[i] = trapz(1/beta[:i], positions[:i]) # radians
    if units == 'deg':
        phases = np.degrees(phases)
    elif units == 'tune':
        phases /= 2*np.pi
    return phases
    
    
def params_from_transfer_matrix(M):
    """Return dictionary of lattice parameters from the transfer matrix."""
    keys = ['frac_tune_x', 'frac_tune_y', 'alpha_x', 'alpha_y', 'beta_x',
            'beta_y', 'gamma_x', 'gamma_y']
    lattice_params = {key: None for key in keys}

    cos_phi_x = (M[0, 0] + M[1, 1]) / 2
    cos_phi_y = (M[2, 2] + M[3, 3]) / 2
    if abs(cos_phi_x) >= 1 or abs(cos_phi_y) >= 1 :
        return lattice_params
    sign_x = sign_y = +1
    if abs(M[0, 1]) != 0:
        sign_x = M[0, 1] / abs(M[0, 1])
    if abs(M[2, 3]) != 0:
        sign_y = M[2, 3] / abs(M[2, 3])
    sin_phi_x = sign_x * np.sqrt(1 - cos_phi_x**2)
    sin_phi_y = sign_y * np.sqrt(1 - cos_phi_y**2)

    nux = sign_x * np.arccos(cos_phi_x) / (2 * np.pi)
    nuy = sign_y * np.arccos(cos_phi_y) / (2 * np.pi)
    beta_x = M[0, 1] / sin_phi_x
    beta_y = M[2, 3] / sin_phi_y
    alpha_x = (M[0, 0] - M[1, 1]) / (2 * sin_phi_x)
    alpha_y = (M[2, 2] - M[3, 3]) / (2 * sin_phi_y)
    gamma_x = -M[1, 0] / sin_phi_x
    gamma_y = -M[3, 2] / sin_phi_y

    lattice_params['frac_tune_x'] = nux
    lattice_params['frac_tune_y'] = nuy
    lattice_params['beta_x'] = beta_x
    lattice_params['beta_y'] = beta_y
    lattice_params['alpha_x'] = alpha_x
    lattice_params['alpha_y'] = alpha_y
    lattice_params['gamma_x'] = gamma_x
    lattice_params['gamma_y'] = gamma_y
    return lattice_params


def get_perveance(kin_energy, mass, line_density):
    """Return the dimensionless space charge perveance.
    
    kin_energy : kinetic energy per particle [GeV]
    mass : mass per particle [GeV/c^2]
    line_density : particles per length [m^-1]
    """
    gamma = 1 + (kin_energy / mass)
    beta = np.sqrt(1 - (1 / (gamma**2)))
    return (2 * classical_proton_radius * line_density) / (beta**2 * gamma**3)