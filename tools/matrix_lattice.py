"""This module contains functions to track particles using transfer matrices.

"""

import numpy as np
import numpy.linalg as la
from . import coupling as BL
from .utils import rotate_vec, rotate_mat, apply, normalize


# Element definitions
def M_drift(L):
    """Drift transfer matrix."""
    M = np.zeros((4, 4))
    M[:2, :2] = M[2:, 2:] = [[1, L], [0, 1]]
    return M
    
    
def M_quad(L, k, kind='qf', tilt=0):
    """Focusing quadrupole transfer matrix."""
    k = np.sqrt(np.abs(k))
    cos = np.cos(k*L)
    sin = np.sin(k*L)
    cosh = np.cosh(k*L)
    sinh = np.sinh(k*L)
    if kind == 'qf':
        M = np.array([[cos, sin/k, 0, 0],
                      [-k*sin, cos, 0, 0],
                      [0, 0, cosh, sinh/k],
                      [0, 0, k*sinh, cosh]])
    elif kind == 'qd':
        M = np.array([[cosh, sinh/k, 0, 0],
                      [k*sinh, cosh, 0, 0],
                      [0, 0, cos, sin/k],
                      [0, 0, -k*sin, cos]])
    if tilt:
        M = rotate_mat(M, np.radians(tilt))
    return M


def fodo(k1, k2, L, fill_fac=0.5, quad_tilt=0, start='quad'):
    """Create simple FODO lattice."""
    Lquad = fill_fac * L / 2
    Ldrift = (1 - fill_fac) * L / 2
    lattice = MatrixLattice()
    if start == 'quad':
        lattice.add(M_quad(0.5*Lquad, k1, 'qf', quad_tilt))
        lattice.add(M_drift(Ldrift))
        lattice.add(M_quad(Lquad, k2, 'qd', -quad_tilt))
        lattice.add(M_drift(Ldrift))
        lattice.add(M_quad(0.5*Lquad, k1, 'qf', quad_tilt))
    elif start == 'drift':
        lattice.add(M_drift(0.5*Ldrift))
        lattice.add(M_quad(Lquad, k1, 'qf', quad_tilt))
        lattice.add(M_drift(Ldrift))
        lattice.add(M_quad(Lquad, k2, 'qd', -quad_tilt))
        lattice.add(M_drift(0.5*Ldrift))
    lattice.analyze()
    return lattice


class MatrixLattice:
    """Lattice representation using transfer matrices."""
    
    def __init__(self):
        self.matrices = [] # [element1, element2, ...]
        self.M = None # transfer matrix
        self.V = None # symplectic normalization matrix
        self.Vinv = None # inverse of V
        self.eigvals = None # eigenvalues of transfer matrix
        self.eigvecs = None # each column is an eigenvector of M
        self.v1 = None # eigenvector 1 (can get other two by complex conjugate)
        self.v2 = None # eigenvector 2
        self.eig1 = None # eigenvalue 1
        self.eig2 = None # eigenvalue 2
        
    def n_elements(self):
        """Return the number of elements in the lattice."""
        return len(self.matrices)
        
    def build(self):
        """Create complete lattice transfer matrix."""
        n_elements = self.n_elements()
        if n_elements == 0:
            return
        if n_elements == 1:
            self.M = self.matrices[0]
        else:
            self.M = la.multi_dot(list(reversed(self.matrices)))
            
    def analyze(self):
        """Compute the lattice parameters."""
        self.eigvals, self.eigvecs_raw = la.eig(self.M)
        self.eigvecs = BL.normalize(self.eigvecs_raw)
        self.eig1, self.eig2 = self.eigvals[[0, 2]]
        self.v1_raw, self.v2_raw = self.eigvecs_raw[[0, 2]]
        self.v1, self.v2 = self.eigvecs[[0, 2]]
        self.V = BL.construct_V(self.eigvecs)
        self.Vinv = la.inv(self.V)
        self._get_twiss2D()
        self._get_twiss4D()
        
    def _get_twiss2D(self):
        """Get the 2D Twiss parameters."""
        M = self.M
        self.params2D = {}
        cos_phi_x = (M[0, 0] + M[1, 1]) / 2
        cos_phi_y = (M[2, 2] + M[3, 3]) / 2
        sign_x = sign_y = +1
        if abs(M[0, 1]) != 0:
            sign_x = M[0, 1] / abs(M[0, 1])
        if abs(M[2, 3]) != 0:
            sign_y = M[2, 3] / abs(M[2, 3])
        sin_phi_x = sign_x * np.sqrt(1 - cos_phi_x**2)
        sin_phi_y = sign_y * np.sqrt(1 - cos_phi_y**2)
        self.params2D['mux'] = sign_x * np.arccos(cos_phi_x)
        self.params2D['muy'] = sign_y * np.arccos(cos_phi_y)
        self.params2D['nux'] = self.params2D['mux'] / (2 * np.pi)
        self.params2D['nuy'] = self.params2D['muy'] / (2 * np.pi)
        self.params2D['bx'] = M[0, 1] / sin_phi_x
        self.params2D['by'] = M[2, 3] / sin_phi_y
        self.params2D['ax'] = (M[0, 0] - M[1, 1]) / (2 * sin_phi_x)
        self.params2D['ay'] = (M[2, 2] - M[3, 3]) / (2 * sin_phi_y)
        
    def _get_twiss4D(self):
        """Get the 4D Twiss parameters."""
        a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y, u, nu1, nu2 = BL.extract_twiss(self.V)
        self.params4D = {'a1x':a1x, 'a1y':a1y, 'a2x':a2x, 'a2y':a2y, 
                         'b1x':b1x, 'b1y':b1y, 'b2x':b2x, 'b2y':b2y, 
                         'u':u, 'nu1':nu1, 'nu2':nu2}
        self.params4D['mu1'] = np.arccos(self.eig1.real)
        self.params4D['mu2'] = np.arccos(self.eig2.real)
        

    def add(self, mat):
        """Add an element to the end of the lattice."""
        self.matrices.append(mat)
        self.build()
        
    def rotate(self, phi):
        """Apply transverse rotation to all elements."""
        self.M = rotate_mat(self.M, np.radians(phi))
        self.analyze()

    def is_stable(self):
        """Return True if all eigvals lie on unit circle in complex plane."""
        for eigval in self.eigvals:
            if abs(la.norm(eigval) - 1) > 1e-5:
                return False
        return True
    
    def normal_form(self):
        """Return normal form of lattice transfer matrix."""
        return la.multi_dot([la.inv(self.V), self.M, self.V])
    
    def fill_eigvecs(self, nparts=50, mode=1):
        """Generate particles distributed uniformly in phase along each or
        both of the transfer matrix eigenvectors."""
        X = []
        phases = np.linspace(0, 2*np.pi, nparts)
        if mode in (1, 'both'):
            for phase in phases:
                X.append(np.real(self.v1 * np.exp(1j*phase)))
        if mode in (2, 'both'):
            for phase in phases:
                X.append(np.real(self.v2 * np.exp(1j*phase)))
        return np.array(X)
    
    def matched_dist(self, nparts=1000, kind='KV', eps1=0.5, eps2=0.5):
        """Generate matched distribution."""
        X_n = normalize(np.random.normal(size=(nparts, 4)))
        A = np.sqrt(np.diag([eps1, eps1, eps2, eps2]))
        X = apply(np.matmul(self.V, A), X_n)
        return X
    
    def track_part(self, x, nturns=1, norm_coords=False):
        """Track a single particle."""
        if norm_coords:
            x = np.matmul(self.Vinv, x)
            M = self.normal_form()
        else:
            M = self.M
        X = [x]
        for _ in range(nturns):
            X.append(np.matmul(M, X[-1]))
        return np.array(X)

    def track_bunch(self, X, nturns=1, norm_coords=False):
        """Track a particle bunch."""
        if norm_coords:
            X = apply(Vinv, X)
            M = self.normal_form()
        else:
            M = self.M
        coords = [X]
        for _ in range(nturns):
            coords.append(apply(M, coords[-1]))
        return np.array(coords)
    
    def print_params(self, kind='2D'):
        """Print the lattice parameters."""
        print('{} lattice parameters'.format(kind))
        print('---------------------')
        params = self.params2D if kind == '2D' else self.params4D
        for name, val in params.items():
            if name in ['mux', 'muy', 'mu1', 'mu2', 'nu1', 'nu2']:
                val = np.degrees(val)
            print('{} = {:.2f}'.format(name, val))
