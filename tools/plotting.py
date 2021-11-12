from cycler import cycler
import copy

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, animation, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from skimage import filters
import seaborn as sns
import scipy

from .beam_analysis import get_ellipse_coords, rms_ellipse_dims
from . import utils


DEFAULT_COLORCYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']


_labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
_labels_norm = [r"$x_n$", r"$x_n'$", r"$y_n$", r"$y_n'$"]
var_indices = {'x':0, 'xp':1, 'y':2, 'yp':3}

    
def colorcycle(cmap, nsamples=1, start_end=(0, 1)):
    """Return color cycle from colormap."""
    start, end = start_end
    colors = [cmap(i) for i in np.linspace(start, end, nsamples)]
    return cycler('color', colors)
    
    
def despine(ax_list, sides=('top', 'right')):
    """Remove the axis spines."""
    # Might want to look at this function in seaborn: https://github.com/mwaskom/seaborn/blob/master/seaborn/utils.py
    if sides == 'all':
        sides = ('top', 'left', 'bottom', 'right')
    if type(sides) is str:
        sides = (sides,)
    for ax in ax_list:
        for side in sides:
            if ax is not None:
                ax.spines[side].set_visible(False)
                if side == 'left':
                    ax.yaxis.set_visible(False)
                if side == 'bottom':
                    ax.xaxis.set_visible(False)
            
            
def make_lower_triangular(axes):
    """"Remove all top-right subplots (above diagonal)."""
    for i, j in zip(*np.triu_indices(axes.shape[0], 1)):
        axes[i, j].axis('off')
        

def get_labels(units=None, norm_labels=False):
    """Return phase space labels ([x, xp, y, yp]), possibly with units."""
    labels = _labels_norm if norm_labels else _labels
    if units == 'm-rad':
        unit_labels = 2 * [' [m]', ' [rad]']
    elif units == 'mm-mrad':
        unit_labels = 2 * [' [mm]', ' [mrad]']
    else:
        unit_labels = 4 * ['']
    return [lab + ulab for (lab, ulab) in zip(labels, unit_labels)]
    
    
def set_labels(ax_list, labels=[], kind='xlabel', **kws):
    """Set the axis labels/titles for the list of subplots."""
    for ax, label in zip(ax_list, labels):
        if kind == 'xlabel':
            ax.set_xlabel(label, **kws)
        elif kind == 'ylabel':
            ax.set_ylabel(label, **kws)
        if kind == 'title':
            ax.set_title(label, **kws)

            
def set_limits(ax_list, limits=[], dim='x'):
    """Set the x and/or y axis limits for the list of subplots.
    
    dim : {'x', 'y', or 'xy'}
    """
    for ax, lim in zip(ax_list, limits):
        if 'x' in dim:
            ax.set_xlim(lim)
        if 'y' in dim:
            ax.set_ylim(lim)
            

def set_ticks(ax_list, ticks=[], dim='x'):
    """Set the same x and/or y axis ticks for the list of subplots.
    
    dim : {'x', 'y', or 'xy'}
    """
    for ax in ax_list:
        if 'x' in dim:
            ax.set_xticks(ticks)
        if 'y' in dim:
            ax.set_yticks(ticks)
            
            
def hide_axis_labels(ax_list, dim='x'):
    """Turn off tick labels and offset text along u axis, where u = x or y"""
    for ax in ax_list:
        uaxis = {'x':ax.xaxis, 'y':ax.yaxis}[dim]
        uaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
        uaxis.offsetText.set_visible(False)
        
        
def toggle_grid(axes_list, switch='off', **kws):
    """Turn grid on or off."""
    grid = {'off':False, 'on':True}[switch]
    for ax in axes_list:
        ax.grid(grid, **kws)
        
        
def remove_annotations(axes):
    """Delete all text (as well as arrows) from the figure."""
    if type(axes) is not np.ndarray:
        axes = np.array([axes])
    for ax in axes.flat:
        for annotation in ax.texts:
            annotation.set_visible(False)
            
        
def set_share_axes(axes, sharex=False, sharey=False, type_if_1D='row'):
    """Custom axis sharing.
    
    Taken from StackOverflow answer by 'herrlich10': 'https://stackoverflow.com/questions/23528477/share-axes-in-matplotlib-for-only-part-of-the-subplots'.

    Parameters
    ----------
    axes : ndarray of Axes objectes
        The axes to group together.
    sharex{y} : bool
        Whether to share the x{y} axis.
    type_if_1D : {'row', 'col'}
        Determines if we are working with a column or row if `axes` is 1D.
    """
    if axes.ndim == 1:
        if type_if_1D == 'row':
            axes = axes[np.newaxis, :]
        elif type_if_1D == 'col':
            axes = axes[:, np.newaxis]
    target = axes.flat[0]
    for ax in axes.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    if sharex:
        hide_axis_labels(axes[:-1, :].flat, 'x')
    if sharey:
        hide_axis_labels(axes[:, 1:].flat, 'y')
        

def process_limits(mins, maxs, pad=0., zero_center=False):
    # Same limits for x/y and x'/y'
    widths = np.abs(mins - maxs)
    for (i, j) in [[0, 2], [1, 3]]:
        delta = 0.5 * (widths[i] - widths[j])
        if delta < 0.:
            mins[i] -= abs(delta)
            maxs[i] += abs(delta)
        elif delta > 0.:
            mins[j] -= abs(delta)
            maxs[j] += abs(delta)
    # Pad the limits by fractional amount `pad`.
    deltas = 0.5 * np.abs(maxs - mins)
    padding = deltas * pad
    mins -= padding
    maxs += padding
    if zero_center:
        maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
        mins = -maxs
    return mins, maxs
    

def auto_limits(X, pad=0., zero_center=False, sigma=None):
    """Determine axis limits from coordinate array."""
    if sigma is None:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        widths = 2.0 * sigma * stds
        mins = means - 0.5 * widths
        maxs = means + 0.5 * widths
    mins, maxs = process_limits(mins, maxs, pad, zero_center)
    return [(lo, hi) for lo, hi in zip(mins, maxs)]
    
    
def auto_limits_global(coords, pad=0., zero_center=False, sigma=None):
    """Determine axis limits from multiple coordinate arrays."""
    if sigma is None:
        mins = np.min([np.min(X, axis=0) for X in coords], axis=0)
        maxs = np.max([np.max(X, axis=0) for X in coords], axis=0)
    else:
        means = np.mean([np.mean(X, axis=0) for X in coords], axis=0)
        stds = np.max([np.std(X, axis=0) for X in coords], axis=0)
        widths = 2.0 * sigma * stds
        mins = means - 0.5 * widths
        maxs = means + 0.5 * widths
    mins, maxs = process_limits(mins, maxs, pad, zero_center)
    return [(lo, hi) for lo, hi in zip(mins, maxs)]
    
    
def max_u_up(X):
    """Get maximum position (u) and slope (u') in the coordinate array.

    X : ndarray, shape (nparts, 4)
        Coordinate array with columns: [x, x', y, y'].
    """
    xmax, xpmax, ymax, ypmax = np.max(X, axis=0)
    umax, upmax = max(xmax, ymax), max(xpmax, ypmax)
    return np.array([umax, upmax])
    
    
def min_u_up(X):
    """Get minimum position (u) and slope (u') in the coordinate array.

    X : ndarray, shape (nparts, 4)
        Coordinate array with columns: [x, x', y, y'].
    """
    xmin, xpmin, ymin, ypmin = np.min(X, axis=0)
    umin, upmin = min(xmin, ymin), max(xpmin, ypmin)
    return np.array([umin, upmin])
    
    
def max_u_up_global(coords):
    """Get the maximum x{y} and x'{y'} extents for any frame in `coords`.

    coords : ndarray, shape (nframes, nparts, 4)
        Coordinate arrays at each frame.
    """
    u_up_local_maxes = np.array([max_u_up(X) for X in coords])
    umax_global, upmax_global = np.max(u_up_local_maxes, axis=0)
    return np.array([umax_global, upmax_global])


def auto_n_bins_4D(X, limits=None):
    """Try to determine best number of bins to use in 2D histograms in corner plot.
    
    We should be able to find an algorithm to do this nicely. (Edit: for our 
    purposes, it is best to run auto-binning on each 1D histogram and use these
    for the 2D grids. This results in grainy plots, but it resembles a scatter
    plot with points shaded by density.)
    """
    raise NotImplementedError
    
    
def pair_grid(
    n_dims, figsize=None, limits=None, space=None, spines=False,
    labels=None, label_kws=None, tick_kws=None
):
    """Create square grid of subplots (Seaborn does this with PairGrid)."""
    constrained_layout = space is None
    fig, axes = plt.subplots(n_dims, n_dims, figsize=figsize, 
                             sharex='col', sharey=False, 
                             constrained_layout=constrained_layout)
    if not constrained_layout:
        fig.subplots_adjust(wspace=space, hspace=space)
    make_lower_triangular(axes)
    if not spines:
        despine(axes.flat, ('top', 'right'))
        despine(axes.diagonal(), 'left')
        
    # Configure axis sharing
    lcol, brow = axes[1:, 0], axes[-1, :]
    for i, row in enumerate(axes[1:, :]): 
        set_share_axes(row[:i+1], sharey=True)
    set_share_axes(axes.diagonal(), sharey=True)
    toggle_grid(axes.diagonal(), 'off')

    # Limits
    if limits is not None:
        set_limits(brow, limits, 'x')
        set_limits(lcol, limits[1:], 'y')

    # Labels
    if label_kws is None:
        label_kws = dict()
    label_kws.setdefault('fontsize', 'medium')
    if labels:
        set_labels(brow, labels, 'xlabel', **label_kws)
        set_labels(lcol, labels[1:], 'ylabel', **label_kws)

    # Ticks
    if tick_kws is None:
        tick_kws  = dict()
    tick_kws.setdefault('labelsize', 'small')
    fig.align_labels()
    for ax in axes.flat:
        ax.tick_params(**tick_kws)
        
    return fig, axes


def pair_grid_nodiag(
    n_dims, figsize=None, limits=None, space=None, spines=False,
    labels=None, label_kws=None, tick_kws=None, 
    constrained_layout=True
):
    """Same as `pair_grid` but without diagonal subplots."""
    fig, axes = plt.subplots(n_dims - 1, n_dims - 1, figsize=figsize, 
                             sharex='col', sharey='row', 
                             constrained_layout=constrained_layout)
    if not constrained_layout:
        fig.subplots_adjust(wspace=space, hspace=space)
    make_lower_triangular(axes)
    if not spines:
        despine(axes.flat, ('top', 'right'))
        
    lcol, brow = axes[:, 0], axes[-1, :]

    # Limits
    if limits is not None:
        set_limits(brow, limits, 'x')
        set_limits(lcol, limits[1:], 'y')

    # Labels
    if label_kws is None:
        label_kws = dict()
    label_kws.setdefault('fontsize', 'medium')
    if labels:
        set_labels(brow, labels, 'xlabel', **label_kws)
        set_labels(lcol, labels[1:], 'ylabel', **label_kws)

    # Ticks
    if tick_kws is None:
        tick_kws  = dict()
    tick_kws.setdefault('labelsize', 'small')
    fig.align_labels()
    for ax in axes.flat:
        ax.tick_params(**tick_kws)
        
    return fig, axes
    

def corner(
    X, kind='hist', figsize=None, limits=None, hist_height_frac=0.6,
    samples=None, smooth_hist=False, thresh=None, blur=None,
    rms_ellipse_kws=None, autolim_kws=None, grid_kws=None, diag_kws=None,
    **plot_kws
):
    """Plot the pairwise relationships between the coordinates.

    This is similar to routines in other packages like `scatter_matrix` in 
    Pandas or `pairplot` in Seaborn.
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Array of d-dimensional coordinates.
    kind : {'hist', 'scatter'}:
        The type of bivariate plot.
    figsize : tuple or int
        Size of the figure (x_size, y_size). 
    limits : list
        List of (min, max) for each dimension.
    hist_height_frac : float
        Fractional reduction of 1D histogram heights.
    samples : int or float
        If an int, the number of points to use in the scatter plots. If a
        float, the fraction of total points to use.
    smooth_hist : bool
        If True, connect 1D histogram heights with lines. Otherwise use a
        bar plot.
    thresh : float
        In the 2D histograms, with count < thresh will not be plotted.
    blur : float
        Apply a Gaussian blur to the 2D histograms with sigma=blur.
    rms_ellipse_kws : dict
        Key word arguments for plotting of the rms ellipses. Pass
        {'2rms': False} to plot the true rms ellipse instead of the 2-rms
        ellipse. If None, the ellipses are not plotted.
    autolim_kws : dict
        Key word arguments for `auto_limits` method.
    grid_kws : dict
        Key word arguments for `pair_grid` method.
    diag_kws : dict
        Key word arguments for the univariate plots.
    plot_kws : dict
        Key word arguments for the bivariate plots. They will go to either
        `scatter` or `hist2d`.
        
    Returns
    -------
    axes : ndarray, shape (d, d)
        Array of subplots.
    
    To do
    -----
    * Option to plot d-dimensional histogram instead of a coordinate array.
    """
    # Default key word arguments.
    if kind =='scatter' or kind == 'scatter_density':
        plot_kws.setdefault('s', 3)
        plot_kws.setdefault('c', 'black')
        if 'color' in plot_kws:
            plot_kws['c'] = plot_kws.pop('color')
        plot_kws.setdefault('marker', '.')
        plot_kws.setdefault('ec', 'none')
        plot_kws.setdefault('zorder', 5)
    elif kind == 'hist':
        plot_kws.setdefault('cmap', 'dusk_r')
        plot_kws.setdefault('shading', 'auto')
        plot_kws.setdefault('bins', 'auto')
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    diag_kws.setdefault('histtype', 'step')
    diag_kws.setdefault('bins', 'auto')
    if autolim_kws is None:
        autolim_kws = dict()

    # Create figure.
    n_parts, n_dims = X.shape
    if figsize is None:
        f = n_dims * 7.5 / 6.0
        figsize = (1.025 * f, f)
    if limits is None:
        limits = auto_limits(X, **autolim_kws)
    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault('labels', ["x [mm]", "x' [mrad]",
                                   "y [mm]", "y' [mrad]",
                                   "z [m]", "dE [MeV]"])
    grid_kws.setdefault('limits', limits)
    grid_kws.setdefault('figsize', figsize)
    fig, axes = pair_grid(n_dims, **grid_kws)
    
    # Compute the covariance matrix. Multiply by four unless told not to.
    if rms_ellipse_kws is not None:
        Sigma = np.cov(X.T)
        rms_ellipse_kws.setdefault('2rms', True)
        if rms_ellipse_kws.pop('2rms'):
            Sigma *= 4.0
        
    # Univariate plots.
    if smooth_hist:
        diag_kws.pop('histtype')
    bins = diag_kws.pop('bins')
    n_bins = []
    for i, ax in enumerate(axes.diagonal()):
        heights, edges = np.histogram(X[:, i], bins, limits[i])
        centers = utils.get_bin_centers(edges)
        n_bins.append(len(edges) - 1)
        if smooth_hist:
            ax.plot(centers, heights, **diag_kws)
        else:
            ax.hist(centers, len(centers), weights=heights, **diag_kws)
        
    # Take random sample.
    idx = np.arange(n_parts)
    if samples is not None and samples < n_parts:
        if type(samples) is float:
            n = int(samples * n_parts)
        else:
            n = samples
        idx = utils.rand_rows(idx, n)
    
    # Bivariate plots.
    if kind == 'hist':
        bins = plot_kws.pop('bins')
    for i in range(1, len(axes)):
        for j in range(i):
            ax = axes[i, j]
            if kind == 'scatter':
                x, y = X[idx, j], X[idx, i]
                ax.scatter(x, y, **plot_kws)
            elif kind == 'hist':
                x, y = X[:, j], X[:, i]
                if bins == 'auto':
                    Z, xedges, yedges = np.histogram2d(
                        x, y, (n_bins[j], n_bins[i]), (limits[j], limits[i]))
                else:
                    Z, xedges, yedges = np.histogram2d(
                        x, y, bins, (limits[j], limits[i]))
                if blur:
                    Z = filters.gaussian(Z, sigma=blur)
                if thresh:
                    Z = np.ma.masked_less_equal(Z, thresh)
                xcenters = utils.get_bin_centers(xedges)
                ycenters = utils.get_bin_centers(yedges)
                ax.pcolormesh(xcenters, ycenters, Z.T, **plot_kws)
            if rms_ellipse_kws is not None:
                rms_ellipses(Sigma, axes=axes, **rms_ellipse_kws)
    
    # Reduce height of 1D histograms. 
    max_hist_height = 0.
    for ax in axes.diagonal():
        max_hist_height = max(max_hist_height, ax.get_ylim()[1])
    max_hist_height /= hist_height_frac
    for ax in axes.diagonal():
        ax.set_ylim(0, max_hist_height)
    return axes
    
    
def corner_env(
    env_params, dims='all', axes=None, figsize=None, limits=None,
    units='mm-mrad', norm_labels=False, fill=False, cmap=None,
    cmap_range=(0, 1), autolim_kws=None, grid_kws=None, fill_kws=None,
    **plt_kws
):
    """Plot projected phase space ellipses from Danilov envelope parameters.
    
    Inputs
    ------
    env_params : ndarray, shape (8,) or (n, 8)
        Envelope parameters [a, b, a', b', e, f, e', f']. If multiple rows are
        provided, each will be plotted as different ellipse.
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ("x", "yp") or (0, 3).
    axes : single axis or (3, 3) array of axes).
        If plotting onto existing axes.
    figsize : tuple or int
        Size of the figure (x_size, y_size). 
    limits : list
        List of (min, max) for each dimension.
    units : str, bool, or None
        Whether to display units on the axis labels. Options are 'mm-mrad' or
        'm-rad'.
    norm_labels : bool
        Whether to add '_n' to the axis labels ('x' -> 'x_n').
    fill : bool
        Whether to fill the ellipses.
    cmap : list of colors, Matplotlib colormap, or str 
        Determines the color cycle if plotting multiple envelopes.
    cmap_range : (min, max)
        The locations for the color cycle to start and end in the color map
        (between 0 and 1).
    autolim_kws : dict
        Key word arguments for `auto_limits` method.
    grid_kws : dict
        Key word arguments for `pair_grid` method.
    fill_kws : dict
        Key word arguments for `ax.fill` if filling the ellipses.
    **plt_kws
        Key word arguments for `ax.plot` if plotting ellipse boundaries. (We
        should eventually be able to pass a list to use different key words
        for each ellipse).
        
    Returns
    -------
    axes : ndarray, shape (3, 3)
        Array of subplots.
    """   
    # Get ellipse boundary data.
    if type(env_params) is not np.ndarray:
        env_params = np.array(env_params)
    if env_params.ndim == 1:
        env_params = env_params[np.newaxis, :]
    coords = [get_ellipse_coords(p, npts=100) for p in env_params]
        
    # Set default key word arguments.
    n_env = len(env_params)
    color = None if n_env > 1 else 'black'
    plt_kws.setdefault('lw', None)
    plt_kws.setdefault('color', color)
    plt_kws.setdefault('zorder', 10)
    if fill_kws is None:
        fill_kws = dict()
    fill_kws.setdefault('lw', 1)
    fill_kws.setdefault('fc', 'lightsteelblue')
    fill_kws.setdefault('ec', 'k')
    fill_kws.setdefault('zorder', 10)
        
    # Configure axes limits.
    if limits is None:
        if autolim_kws is None:
            autolim_kws = dict()
        autolim_kws.setdefault('pad', 0.5)
        limits = auto_limits_global(coords, **autolim_kws)
            
    # Create figure.
    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault('figsize', figsize)
    grid_kws.setdefault('limits', limits)
    grid_kws.setdefault('constrained_layout', True)
    grid_kws.setdefault('labels', get_labels(units, norm_labels))
    if dims == 'all':
        if axes is None:
            n_dims = 4
            fig, axes = pair_grid_nodiag(n_dims, **grid_kws)
    else:
        if axes is None:
            fig, ax = plt.subplots(
                figsize=figsize,
                constrained_layout=grid_kws['constrained_layout'],
            )
            i, j = dims
            if type(i) is str:
                i = var_indices[i]
            if type(j) is str:
                j = var_indices[j]
            ax.set_xlabel(labels[i], **label_kws)
            ax.set_ylabel(labels[j], **label_kws)
            ax.set_xlim(limits[i])
            ax.set_ylim(limits[j])
        else:
            ax = axes
    
    # Set the color cycle.
    colors = None
    if n_env > 1 and cmap is not None:
        if type(cmap) in [tuple, list]:
            colors = cmap
        else:
            if type(cmap) is str:
                cmap = matplotlib.cm.get_cmap(cmap)
            start, end = cmap_range
            colors = [cmap(i) for i in np.linspace(start, end, n_env)]
        color_cycle = cycler('color', colors)
        if dims != 'all':
            ax.set_prop_cycle(color_cycle)
        else:
            for ax in axes.flat:
                ax.set_prop_cycle(color_cycle)

    # Plot
    for X in coords:
        if dims != 'all':
            j, i = [var_indices[dim] for dim in dims]
            if fill:
                ax.fill(X[:, j], X[:, i], **fill_kws)
            else:
                ax.plot(X[:, j], X[:, i], **plt_kws)
        else:
            for i in range(3):
                for j in range(i + 1):
                    ax = axes[i, j]
                    if fill:
                        ax.fill(X[:, j], X[:, i+1], **fill_kws)
                    else:
                        ax.plot(X[:, j], X[:, i+1], **plt_kws)
    return axes
    
    
def fft(ax, x, y):
    """Compute and plot the FFT of two signals x and y on the same figure.
    
    Let N be the number of samples, M=N/2, x(t)=signal as function of time, 
    and w(f) be the FFT of x. Then w[0] is zero frequency component, w[1:M] 
    are positive frequency components, and w[M:N] are negative frequency 
    components.
    """
    N = len(x)
    M = N // 2
    f = (1/N) * np.arange(M)
    xf = (1/M) * np.abs(scipy.fft.fft(x)[:M])
    yf = (1/M) * np.abs(scipy.fft.fft(y)[:M])
    ax.set_xlabel('Tune')
    ax.set_ylabel('Amplitude')
    ax.plot(f[1:], xf[1:], label=r'$\nu_x$')
    ax.plot(f[1:], yf[1:], label=r'$\nu_y$')
    return ax


def scatter_density(ax, x, y, samples=None, sort=True, bins=40,
                    method='interp', **kws):
    """Scatter plot with colors weighted by local density.
    
    Note: it is easier to just plot a 2D histogram and set the `vmin` parameter 
    to something slightly nonzero. 
    
    Taken from StackOverflow: 'https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib'.
    
    Parameters
    ----------
    ax : AxesSubplot
        The axis on which to plot.
    x, y : ndarray
        The data.
    samples : int
        The number of random samples to plot. All points are used in the
        density estimates. If None, plot all the points.
    sort : bool
        If True, plot higher-density points on top.
    bins : int
        If interpolating from 2d histogram, this determines the number of bins
        used in the histogram.
    method : {'interp', 'kde'}
        If 'interp', interpolate the densities from a 2d histogram using a
        spline method. If 'kde', use a gaussian kernel density estimate. KDE
        is MUCH slower.
    **kws
        Key word arguments passed to matplotlib.pyplot.scatter.
    """
    xy = np.vstack([x, y])
    if method == 'kde':
        z = scipy.stats.gaussian_kde(xy)(xy)
    elif method == 'interp':
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = scipy.interpolate.interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            xy.T,
            method='splinef2d',
            bounds_error=False
        )
    else:
        raise ValueError("'method' must be in ['interp', 'kde']")

    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
    if samples is not None and len(z) > samples:
        idx = np.random.choice(len(z), samples, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        
    ax.scatter(x, y, c=z, **kws)
    return ax


def vector(ax, v, origin=(0, 0), color='black', lw=None, 
           style='->', head_width=0.4, head_length=0.8):
    """Plot 2D vector `v` as an arrow."""
    props = dict()
    props['arrowstyle'] = '{},head_width={},head_length={}'.format(
        style, head_width, head_length
    )
    props['shrinkA'] = props['shrinkB'] = 0
    props['fc'] = props['ec'] = color
    props['lw'] = lw
    ax.annotate('', xy=(origin[0]+v[0], origin[1]+v[1]),
                xytext=origin, arrowprops=props)
    return ax


def eigvec_trajectory(ax, M, dim1='x', dim2='y', colors=('red', 'blue'), turns=20,
                      arrow_kws=None, scatter_kws=None):
    """Plot the trajectory of the eigenvectors of the transfer matrix.
    
    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
        The axis on which to plot.
    M : ndarray, shape (4, 4)
        The lattice transfer matrix.
    dim1{dim2} : str or int
        The component of the eigenvector to plot on the x{y} axis. Can be
        either {'x', 'xp', 'y', 'yp'} or {0, 1, 2, 3}.
    colors : (color1, color2)
        The colors to use for eigenvector 1 and 2.
        
    Returns
    -------
    ax : matplotlib.pyplot.axes object
        The modified axis.
    """
    def track(vector, M, nturns=1):
        X = [vector]
        for i in range(nturns):
            X.append(np.matmul(M, X[i]))
        return np.array(X)
    
    if type(dim1) is str:
        dim1 = var_indices[dim1]
    if type(dim2) is str:
        dim2 = var_indices[dim2]
    i, j = dim1, dim2
        
    if scatter_kws is None:
        scatter_kws = dict()
    if arrow_kws is None:
        arrow_kws = dict()
        
    eigvals, eigvecs = np.linalg.eig(M)
    v1, _, v2, _ = eigvecs.T
    for v, color in zip((v1, v2), colors):
        X = track(v.real, M, turns)
        ax.scatter(X[:, i], X[:, j], color=color, **scatter_kws)
        vector(ax, v[[i, j]].real, color=color, **arrow_kws)
    return ax


def unit_circle(ax, **kws):
    """Plot the unit circle in the background on the axis."""
    kws.setdefault('zorder', 0)
    kws.setdefault('color', 'k')
    kws.setdefault('ls', '--')
    psi = np.linspace(0, 2*np.pi, 50)
    ax.plot(np.cos(psi), np.sin(psi), **kws)
    return ax


def eigvals_complex_plane(ax, eigvals, colors=('r','b'), legend=True, **kws):
    """Plot the eigenvalues in the complex plane."""
    unit_circle(ax)
    c1, c2 = colors
    ax.scatter(eigvals.real, eigvals.imag, c=[c1, c1, c2, c2], **kws)
    xmax = 1.75
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((-xmax, xmax))
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    if legend:
        mu1, mu2 = np.degrees(np.arccos(eigvals[[0, 2]].real))
        lines = [Line2D([0], [0], marker='o', lw=0, color=c1, ms=2),
                        Line2D([0], [0], marker='o', lw=0, color=c2, ms=2)]
        labels = [r'$\mu_1 = {:.2f}\degree$'.format(mu1),
                  r'$\mu_2 = {:.2f}\degree$'.format(mu2)]
        ax.legend(lines, labels, loc='upper right', handletextpad=0.1,
                  fontsize='small', framealpha=1)
    return ax
    

def ellipse(ax, c1, c2, angle=0.0, center=(0, 0), **plt_kws):
    """Plot ellipse with semi-axes `c1` and `c2`. Angle is given in radians
    and is measured below the x axis."""
    plt_kws.setdefault('fill', False)
    return ax.add_patch(Ellipse(center, 2*c1, 2*c2, -np.degrees(angle), **plt_kws))


def rms_ellipses(Sigmas, figsize=(5, 5), pad=0.5, axes=None, 
                 cmap=None, cmap_range=(0, 1), centers=None, **plt_kws):
    """Plot rms ellipse parameters directly from covariance matrix."""
    Sigmas = np.array(Sigmas)
    if Sigmas.ndim == 2:
        Sigmas = Sigmas[np.newaxis, :, :]
    if axes is None:
        x2_max, y2_max = np.max(Sigmas[:, 0, 0]), np.max(Sigmas[:, 2, 2])
        xp2_max, yp2_max = np.max(Sigmas[:, 1, 1]), np.max(Sigmas[:, 3, 3])
        umax = (1 + pad) * np.sqrt(max(x2_max, y2_max))
        upmax = (1 + pad) * np.sqrt(max(xp2_max, yp2_max))
        limits = 2 * [(-umax, umax), (-upmax, upmax)]
        fig, axes = pair_grid_nodiag(4, figsize, limits, constrained_layout=False)

    colors = None
    if len(Sigmas) > 1 and cmap is not None:
        start, end = cmap_range
        colors = [cmap(i) for i in np.linspace(start, end, len(Sigmas))]
        
    if centers is None:
        centers = 4 * [0.0]
    
    dims = {0:'x', 1:'xp', 2:'y', 3:'yp'}
    for l, Sigma in enumerate(Sigmas):
        for i in range(3):
            for j in range(i + 1):
                angle, c1, c2 = rms_ellipse_dims(Sigma, dims[j], dims[i + 1])
                if colors is not None:
                    plt_kws['color'] = colors[l]
                xcenter = centers[j]
                ycenter = centers[i + 1]
                ellipse(axes[i, j], c1, c2, angle, center=(xcenter, ycenter), **plt_kws)
    return axes
