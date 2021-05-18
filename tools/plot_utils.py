import numpy as np


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

