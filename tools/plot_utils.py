import numpy as np


PHASE_SPACE_LABELS = [r"x", r"x'", r"y", r"y'", r"z", r"dE"]
PHASE_SPACE_LABELS_UNITS = [r"x [mm]", r"x' [mrad]", r"y [mm]", r"y' [mrad]", r"z [m]", r"dE [MeV]"]


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    
    Found here: https://stackoverflow.com/questions/18311909/how-do-i-annotate-with-power-of-ten-formatting
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r'${0:.{2}f} \cdot 10^{{{1:d}}}$'.format(coeff, exponent, precision)


def moment_label(i, j):
    """Return the label corresponding to Sigma[i, j].
    
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


def moment_label_string(i, j):
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

