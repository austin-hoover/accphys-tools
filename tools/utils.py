"""Various utility functions."""

# Standard
import os
# Third party
import numpy as np
import scipy
import pandas as pd
import sympy
import IPython
from numpy import linalg as la
from sympy import pprint, Matrix
from scipy.integrate import trapz
from IPython.display import display, HTML


# File processing
def list_files(dir):
    """List all files in directory not starting with '.'"""
    files = os.listdir(dir)
    return [file for file in files if not file.startswith('.')]
    
    
def is_empty(dir):
    """Return True if directory is empty."""
    return len(list_files(dir)) > 0
    
    
def delete_files_not_folders(dir):
    """Delete all files in directory and subdirectories."""
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not file.startswith('.'):
                os.remove(os.path.join(root, file))
                
                
def file_exists(file):
    """Return True if the file exists."""
    return os.path.isfile(file)
    
    
# Lists and dicts
def merge_lists(x, y):
    """Returns [x[0], y[0], ..., x[-1], y[-1]]"""
    return [x for pair in zip(a, b) for x in pair]
    
    
def merge_dicts(*dictionaries):
    """Given any number of dictionaries, shallow copy and merge into a new dict.
    
    Precedence goes to key value pairs in latter dictionaries. This function
    will work in Python 2 or 3. Note that in version 3.5 or greater we can just
    call `z = {**x, **y}`, and in 3.9 we can call `z = x | y`, to merge two
    dictionaries (with the values of y replacing those in x).
    
    Example usage:
        >> w = dict(a=1, b=2, c=3)
        >> x = dict(e=4, f=5, c=6)
        >> y = dict(g=7, h=8, f=7)
        >> merge_dicts(w, x, y)
        >> {'a': 1, 'b': 2, 'c': 6, 'e': 4, 'f': 7, 'g': 7, 'h': 8}
    
    Copied from the accepted answer here: 'https://stackoverflow.com/questions//38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o'.
    """
    result = {}
    for dictionary in dictionaries:
        result.update(dictionary)
    return result
    
    
# Display
def tprint(string, indent=4):
    """Print with indent."""
    print(indent*' ' + str(string))
             
             
def show(V, name=None, dec=3):
    """Pretty print matrix with rounded entries."""
    if name:
        print(name, '=')
    pprint(Matrix(np.round(V, dec)))
    
    
def play(anim):
    """Display matplotlib animation. For use in Jupyter notebook."""
    display(HTML(anim.to_jshtml()))
    
    
# NumPy arrays
def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def normalize(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x/la.norm(x), 1, X)


def symmetrize(M):
    """Return a symmetrized version of M.
    
    M : A square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())
    
    
def rand_rows(X, n):
    """Return n random elements of X."""
    nrows = X.shape[0]
    if n >= nrows:
        return X
    idx = np.random.choice(X.shape[0], n, replace=False)
    return X[idx, :]
    
    
# General math
def cov2corr(cov_mat):
    """Convert covariance matrix to correlation matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = la.inv(D)
    corr_mat = la.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat
    
    
def rotation_matrix(phi):
    """2D rotation matrix (cw)""".
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])


# Accelerator physics
def rotation_matrix_4D(phi):
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, 0, S, 0], [0, C, 0, S], [-S, 0, C, 0], [0, -S, 0, C]])


def phase_adv_matrix(mu1, mu2):
    R = np.zeros((4, 4))
    R[:2, :2] = rotation_matrix(mu1)
    R[2:, 2:] = rotation_matrix(mu2)
    return R
    
    
def rotate_vec(x, phi):
    return np.matmul(rotation_matrix_4D(phi), x)


def rotate_mat(M, phi):
    R = rotation_matrix_4D(phi)
    return la.multi_dot([R, M, R.T])
    
    
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


def Vmat_2D(alpha_x, beta_x, alpha_y, beta_y):
    """Normalization matrix (uncoupled)"""
    def V_uu(alpha, beta):
        return np.array([[beta, 0.0], [-alpha, 1.0]]) / np.sqrt(beta)
    V = np.zeros((4, 4))
    V[:2, :2] = V_uu(alpha_x, beta_x)
    V[2:, 2:] = V_uu(alpha_y, beta_y)
    return V
    
    
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


def get_moments_key(i, j):
    """Return the key corresponding to Sigma[i, j].
    
    These keys are used for the column names when creating a DataFrame for
    the beam moments.
    """
    dictionary = {
        (0, 0):'x2' ,
        (1, 0):'xxp',
        (2, 0):'xy',
        (3, 0):'xyp',
        (1, 1):'xp2',
        (2, 1):'yxp',
        (3, 1):'xpyp',
        (2, 2):'y2',
        (3, 2):'yyp',
        (3, 3):'yp2'
    }
    if i < j:
        i, j = j, i
    return dictionary[(i, j)]


def get_moments_label(i, j):
    """Return a string corresponding to Sigma[i, j], e.g. '<x^2>.'"""
    dictionary = {
        (0, 0):r"$\langle{x^2}\rangle$",
        (1, 0):r"$\langle{xx'}\rangle$",
        (2, 0):r"$\langle{xy}\rangle$",
        (3, 0):r"$\langle{xy'}\rangle$",
        (1, 1):r"$\langle{x'^2}\rangle$",
        (2, 1):r"$\langle{yx'}\rangle$",
        (3, 1):r"$\langle{x'y'}\rangle$",
        (2, 2):r"$\langle{y^2}\rangle$",
        (3, 2):r"$\langle{yy'}\rangle$",
        (3, 3):r"$\langle{y'^2}\rangle$"
    }
    str_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}
    if type(i) is str:
        i = str_to_int[i]
    if type(j) is str:
        j = str_to_int[j]
    {'x':0, 'xp':1, 'y':2, 'yp':3}
    if i < j:
        i, j = j, i
    return dictionary[(i, j)]
