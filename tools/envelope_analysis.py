import numpy as np


def get_ellipse_coords(env_params, nparts=50):
    """Get (x, y) coordinates along ellipse boundary from envelope parameters.
    
    Parameters
    ----------
    params : array-like
        The envelope parameters [a, b, a', b', e, f, e', f'].
    n_angles : float
        Number of values of psi to calculate.
        
    Returns
    -------
    coords : ndarray, shape (n_angles, 4)
        Columns are [x, x', y, y'], rows are different values of psi.
    """
    a, b, ap, bp, e, f, ep, fp = env_params
    psi = np.linspace(0, 2 * np.pi, nparts)
    cos, sin = np.cos(psi), np.sin(psi)
    x, xp = a*cos + b*sin, ap*cos + bp*sin
    y, yp = e*cos + f*sin, ep*cos + fp*sin
    coords = np.vstack([x, xp, y, yp]).T
    return coords
