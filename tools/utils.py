import os

import numpy as np
import pandas as pd
import sympy
import IPython
from numpy import linalg as la
from sympy import pprint, Matrix
from IPython.display import display, HTML


# File processing
def list_files(path, join=True):
    files = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and not file.startswith('.'):
            if join:
                files.append(file_path)
            else:
                files.append(file)
    return files
    
    
def is_empty(path):
    return len(list_files(path)) > 0
    
    
def delete_files_not_folders(path):
    for root, folders, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))
                
                
def file_exists(file):
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
    
    
def play(anim, center=True):
    """Display matplotlib animation using HTML."""
    html_string = anim.to_jshtml()
    if center:
        html_string = ''.join(['<center>', html_string, '<center>'])
    display(HTML(html_string))
    
    
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
    Xsamp = np.copy(X)
    if n < X.shape[0]:
        idx = np.random.choice(Xsamp.shape[0], n, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp
    

# Math
def cov2corr(cov_mat):
    """Convert covariance matrix to correlation matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = la.inv(D)
    corr_mat = la.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat
    
    
def rotation_matrix(phi):
    """2D rotation matrix (cw)."""
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])