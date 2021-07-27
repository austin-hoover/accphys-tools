"""
Methods for 4D emittance measurement using the quadrupole scan method.
"""
import sys

import numpy as np
from scipy import optimize as opt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import proplot as plot

from . import utils
from . import plotting as myplt
from . import accphys_utils


# Helper functions
#-------------------------------------------------------------------------------
def _line_to_float_list(line):
    """Convert string to list of floats.
    
    '1 2 3' -> [1.0, 2.0, 3.0])
    """
    return [float(string) for string in line.split()] 


def _lines_to_float_array(lines):
    """Convert list of strings to ndarray of floats.
    
    ['1 2 3', '4 5 6'] -> array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    """
    return np.array([_line_to_float_list(line) for line in lines])


# Reconstruction
#-------------------------------------------------------------------------------
def to_mat(sigma):
    """Return covariance matrix from 10 element moment vector."""
    s11, s22, s12, s33, s44, s34, s13, s23, s14, s24 = sigma
    return np.array([[s11, s12, s13, s14], 
                     [s12, s22, s23, s24], 
                     [s13, s23, s33, s34], 
                     [s14, s24, s34, s44]])


def to_vec(Sigma):
    """Return 10 element moment vector from covariance matrix."""
    s11, s12, s13, s14 = Sigma[0, :]
    s22, s23, s24 = Sigma[1, 1:]
    s33, s34 = Sigma[2, 2:]
    s44 = Sigma[3, 3]
    return np.array([s11, s22, s12, s33, s44, s34, s13, s23, s14, s24])


def reconstruct(transfer_mats, moments, **kwargs):
    """Reconstruct covariance matrix from wire-scanner data.
    
    Parameters
    ----------
    transfer_mats : list of (4, 4) ndarray, length n
        List of 4x4 transfer matrix at each scan.
    moments : list or ndarray, shape (n, 3)
        List of [<xx>, <yy>, <xy>] moments for each scan.
    **kwargs
        Key word arguments passed to scipy.optimize.lsq_linear

    Returns
    -------
    ndarray, shape (4, 4)
        Covariance matrix at reconstruction point.
    """
    A, b = [], []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
        A.append([M[0, 0]**2, M[0, 1]**2, 2*M[0, 0]*M[0, 1], 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, M[2, 2]**2, M[2, 3]**2, 2*M[2, 2]*M[2, 3], 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, M[0, 0]*M[2, 2],  M[0, 1]*M[2, 2],  M[0, 0]*M[2, 3],  M[0, 1]*M[2, 3]])
        b.append(sig_xx)
        b.append(sig_yy)
        b.append(sig_xy)
    A, b = np.array(A), np.array(b)
    lb = 10 * [-np.inf]
    lb[0] = lb[1] = lb[3] = lb[4] = 0.0 # Squared moments can't be negative
    result = opt.lsq_linear(A, b, bounds=(lb, np.inf), **kwargs)
    return to_mat(result.x)


def get_sig_xy(sig_xx, sig_yy, sig_uu, diag_wire_angle):
    """Compute cov(x, y) from horizontal, vertical, and diagonal wires.
    
    [To do: Write formula in terms of cov(x, y) in terms of cov(x, x), 
     cov(y, y), and cov(u, u).]
    """
    phi = np.radians(90.0) + diag_wire_angle
    sin, cos = np.sin(phi), np.cos(phi)
    sig_xy = (sig_uu - sig_xx*(cos**2) - sig_yy*(sin**2)) / (2 * sin * cos)
    return sig_xy


# PTA file data processing
#-------------------------------------------------------------------------------
class Stat:
    """Container for a statistical signal parameter.
    
    Attributes
    ----------
    name : str
        Parameter name.
    rms, fit : float
        Parameter value from rms/Gaussian fit.
    """
    def __init__(self, name, rms, fit):
        self.name, self.rms, self.fit = name, rms, fit

class Signal:
    """Container for profile signal.
    
    Attributes
    ----------
    pos : list
        Wire positions.
    raw : list
        Raw signal amplitudes at each position.
    fit : list
        Gaussian fit amplitudes at each position.
    stats : dict
        Each key is a different statistical parameter: ('Area', 'Mean', ...). 
        Each value is a Stat object that holds the parameter name, rms value, 
        and Gaussian fit value.
    """
    def __init__(self, pos, raw, fit, stats):
        self.pos, self.raw, self.fit, self.stats = pos, raw, fit, stats
        
         
class Profile:
    """Store data from single wire-scanner.
    
    Attributes
    ----------
    hor, ver, dia : Signal
        Signal for horizontal, vertical and diagonal wire.
    """
    def __init__(self, pos, raw, fit=None, stats=None):
        xpos, ypos, upos = pos
        xraw, yraw, uraw = raw
        if fit is None:
            xfit = yfit = ufit = None
        else:
            xfit, yfit, ufit = fit
        if stats is None:
            xstats = ystats = ustats = None
        else:
            xstats, ystats, ustats = stats   
        self.hor = Signal(xpos, xraw, xfit, xstats)
        self.ver = Signal(ypos, yraw, yfit, ystats)
        self.dia = Signal(upos, uraw, ufit, ustats)
        
        
def read_pta_ws(filename):
    """Return dictionary of Profiles from PTA wire-scanner file.
    
    The dictionary keys are the wire-scanner names.
    """
    # Collect lines corresponding to each wire-scanner
    file = open(filename, 'r')
    lines, ws_id = dict(), None
    for line in file:
        line = line.rstrip()
        if line.startswith('RTBT_Diag'):
            ws_id = line.split(':')[-1]
            continue
        if ws_id:
            lines.setdefault(ws_id, []).append(line)
    file.close()
        
    # Read the lines
    profiles = dict()
    for ws_id in sorted(list(lines)):
        # Split lines into three sections:
        #     stats: statistical signal parameters;
        #     raw: wire positions and raw signal amplitudes;
        #     fit: wire positions and Gaussian fit amplitudes.
        # There is one blank line after each section. 
        lines_stats, lines_raw, lines_fit = utils.split_list(lines[ws_id], '')[:3]

        # Remove headers and dashed lines beneath headers.
        lines_stats = lines_stats[2:]
        lines_raw = lines_raw[2:]
        lines_fit = lines_fit[2:]   
        
        # The columns of the following array are ['pos', 'yraw', 'uraw', 'xraw', 
        # 'xpos', 'ypos', 'upos']. (NOTE: This is not the order that is written
        # in the file header.)
        data_arr_raw = _lines_to_float_array(lines_raw)
        pos, yraw, uraw, xraw, xpos, ypos, upos = data_arr_raw.T
        
        # This next array is the same, but it contains 'yfit', 'ufit', 'xfit', 
        # instead of 'yraw', 'uraw', 'xraw'.
        data_arr_fit = _lines_to_float_array(lines_fit)
        pos, yfit, ufit, xfit, xpos, ypos, upos = data_arr_fit.T
                
        # Get statistical signal parameters. (Headers don't give true ordering.)
        xstats, ystats, ustats = dict(), dict(), dict()
        for line in lines_stats:
            tokens = line.split()
            name = tokens[0]
            vals = [float(val) for val in tokens[1:]]
            s_yfit, s_yrms, s_ufit, s_urms, s_xfit, s_xrms = vals
            xstats[name] = Stat(name, s_xrms, s_xfit)
            ystats[name] = Stat(name, s_yrms, s_yfit)
            ustats[name] = Stat(name, s_urms, s_ufit)
        
        profile = Profile([xpos, ypos, upos], 
                          [xraw, yraw, uraw], 
                          [xfit, yfit, ufit], 
                          [xstats, ystats, ustats])
        profiles[ws_id] = profile

    return profiles


def read_pta_harp(filename):
    """Return dictionary of Profiles from PTA harp file.
    
    The dictionary keys are the wire-scanner names.
    """
    file = open(filename, 'r')
    data = []
    for line in file:
        tokens = line.rstrip().split()
        if not tokens or tokens[0] in ['start', 'RTBT_Diag:Harp30', 'PVLoggerID']:
            continue
        data.append([float(token) for token in tokens])
    file.close()
    xpos, xraw, ypos, yraw, upos, uraw = np.array(data).T
    profile = Profile([xpos, ypos, upos], 
                      [xraw, yraw, uraw])
    return profile



# Plotting
#-------------------------------------------------------------------------------
def plot_profiles(measurements, ws_ids, fit=False, kws_raw=None, 
                  kws_fit=None, width=8, vscale=1.4):
    if kws_raw is None:
        kws_raw = dict()
    if kws_fit is None:
        kws_fit = dict()
    kws_raw['legend'] = kws_fit['legend'] = False
    kws_raw.setdefault('marker', '.')
    kws_raw.setdefault('ms', 3)
    kws_raw.setdefault('lw', 0)
    kws_fit.setdefault('color', 'k')
    kws_fit.setdefault('alpha', 0.2)
    kws_fit.setdefault('zorder', 0)
    n_meas = len(measurements)
    fig, axes = plot.subplots(nrows=n_meas, ncols=3, spanx=False, 
                              figsize=(width, vscale*n_meas))
    for i, profiles in enumerate(measurements):
        for ws_id in ws_ids:
            profile = profiles[ws_id]
            axes[i, 0].plot(profile.hor.pos, profile.hor.raw, **kws_raw)
            axes[i, 1].plot(profile.ver.pos, profile.ver.raw, **kws_raw)
            axes[i, 2].plot(profile.dia.pos, profile.dia.raw, **kws_raw)
            if fit:
                axes[i, 0].plot(profile.hor.pos, profile.hor.fit, **kws_fit)
                axes[i, 1].plot(profile.ver.pos, profile.ver.fit, **kws_fit)
                axes[i, 2].plot(profile.dia.pos, profile.dia.fit, **kws_fit)
                
    axes[0, -1].legend(labels=ws_ids, fontsize='small', loc=(1.02, 0), ncols=1)
    axes.format(ylabel='Signal', grid=False, 
                toplabels=['Horizontal', 'Vertical', 'Diagonal'])
    axes[-1, 0].set_xlabel('x [mm]')
    axes[-1, 1].set_xlabel('y [mm]')
    axes[-1, 2].set_xlabel('u [mm]')
    for ax in axes:
        ax.grid(axis='y')
    return axes


def plot_reconstructed_phasespace(transfer_mats, moments, Sigma, Sigma_exp=None, 
                                  twiss=None, scale=2.0):
    
    fig, axes = plot.subplots(ncols=2, figsize=(6, 2.5), 
                              sharex=False, sharey=False, )

    # Plot reconstructed ellipses
    if twiss is not None:
        Sigma = accphys_utils.normalized_Sigma(Sigma, *twiss)
    angle_xxp, cx, cxp = myplt.rms_ellipse_dims(Sigma, 'x', 'xp')
    angle_yyp, cy, cyp = myplt.rms_ellipse_dims(Sigma, 'y', 'yp')
    myplt.ellipse(axes[0], 2 * cx, 2 * cxp, angle_xxp, lw=2)
    myplt.ellipse(axes[1], 2 * cy, 2 * cyp, angle_yyp, lw=2)

    # Plot design ellipses
    plot_exp = Sigma_exp is not None
    if plot_exp:
        if twiss is not None:
            Sigma_exp = accphys_utils.normalized_Sigma(Sigma_exp, *twiss)
        angle_xxp_exp, cx_exp, cxp_exp = myplt.rms_ellipse_dims(Sigma_exp, 'x', 'xp')
        angle_yyp_exp, cy_exp, cyp_exp = myplt.rms_ellipse_dims(Sigma_exp, 'y', 'yp')
        myplt.ellipse(axes[0], 2 * cx_exp, 2 * cxp_exp, angle_xxp_exp, lw=1, alpha=0.25)
        myplt.ellipse(axes[1], 2 * cy_exp, 2 * cyp_exp, angle_yyp_exp, lw=1, alpha=0.25)
        
    # Plot possible points at reconstruction location as lines. 
    def plot_lines(ax, transfer_mats, moments, dim='x', twiss=None, **plt_kws):  
        for transfer_mat, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
            h_pts, v_pts = accphys_utils.possible_points(transfer_mat, sig_xx, sig_yy, dim, twiss)
            ax.plot(h_pts, v_pts, **plt_kws)
        
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    kws = dict(lw=0.3)
    ws_ids = list(transfer_mats)
    for ws_id, color in zip(ws_ids, colors):
        plot_lines(axes[0], transfer_mats[ws_id], moments[ws_id], 'x', twiss, color=color, **kws)
        plot_lines(axes[1], transfer_mats[ws_id], moments[ws_id], 'y', twiss, color=color, **kws)

    # Legends
    custom_lines_1 = [Line2D([0], [0], color='black', lw=2)]
    labels_1 = ['calc']
    if plot_exp:
        custom_lines_1.append(Line2D([0], [0], color='black', lw=1, alpha=0.25))
        labels_1.append('exp')
    custom_lines_2 = [Line2D([0], [0], color=color) for color in colors[:4]]
    axes[1].legend(custom_lines_1, labels_1, ncols=1, loc=(1.02, 0.8), fontsize='small')
    axes[1].legend(custom_lines_2, ws_ids, ncols=1, loc=(1.02, 0), fontsize='small')

    # Formatting
    axes.format(grid=False, xlabel_kw=dict(fontsize='large'), ylabel_kw=dict(fontsize='large'))
    if twiss:
        axes.format(aspect=1)
        labels = [r"$x_n$ [mm]", r"$x'_n$ [mrad]", r"$y_n$ [mm]", r"$y'_n$ [mrad]"]
    else:
        labels = ["x [mm]", "x' [mrad]", "y [mm]", "y' [mrad]"]
        
    max_coords = 2.0 * np.sqrt(np.diag(Sigma))
    xmax, xpmax, ymax, ypmax = scale * max_coords
    axes[0].format(xlim=(-xmax, xmax), ylim=(-xpmax, xpmax), xlabel=labels[0], ylabel=labels[1])
    axes[1].format(xlim=(-ymax, ymax), ylim=(-ypmax, ypmax), xlabel=labels[2], ylabel=labels[3])
    return axes