"""
TO DO
* Redo `corner` method.
* Redo `set_corner_axes` method.
"""
from cycler import cycler
import copy

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, animation, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import seaborn as sns
import scipy

from .beam_analysis import get_ellipse_coords, rms_ellipse_dims
from . import utils


DEFAULT_COLORCYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']


_labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
_labels_norm = [r"$x_n$", r"$x_n'$", r"$y_n$", r"$y_n'$"]
var_indices = {'x':0, 'xp':1, 'y':2, 'yp':3}


def save(figname, dir, **kwargs):
    """Save figure to directory `dir`."""
    if not dir.endswith('/'):
        dir += '/'
    filename = ''.join([dir, figname, '.png'])
    plt.savefig(filename, facecolor='white', **kwargs)


def max_u_up(X):
    """Get maximum position (u) and slope (u') in coordinate array.

    X : ndarray, shape (nparts, 4)
        Coordinate array with columns: [x, x', y, y'].
    """
    xmax, xpmax, ymax, ypmax = np.max(X, axis=0)
    umax, upmax = max(xmax, ymax), max(xpmax, ypmax)
    return np.array([umax, upmax])
    
    
def min_u_up(X):
    """Get minimum position (u) and slope (u') in coordinate array.

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
    # Pad the limits.
    deltas = 0.5 * np.abs(maxs - mins) * pad
    mins -= deltas
    maxs += deltas
    if zero_center:
        maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
        mins = -maxs
    return mins, maxs
    

def auto_limits(X, pad=0., zero_center=False):
    """Determine axis limits from coordinate array."""
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    mins, maxs = process_limits(mins, maxs, pad, zero_center)
    return [(lo, hi) for lo, hi in zip(mins, maxs)]


def auto_n_bins_4D(X, limits=None):
    """Try to determine best number of bins to use in 2D histograms in corner plot.
    
    We should be able to find an algorithm to do this nicely.
    """
    raise NotImplementedError
    
    
# def corner(
#     X, env_params=None, moments=False, limits=None, zero_center=False,
#     samples=None, pad=0., figsize=None, dims='all', kind='scatter',
#     diag_kind='hist', hist_height=0.6, units='mm-mrad', norm_labels=False,
#     text=None, ax=None, diag_kws=None, env_kws=None, text_kws=None, **plt_kws
# ):
#     """Plot the pairwise relationships between the beam phase space coordinates.
    
#     Of course other routines exist such as `scatter_matrix` in Pandas or
#     `pairplot` in Seaborn; however, I could not find an option to only plot
#     a small sample of the particles in the scatter plots while keeping all
#     the particles for the histogram calculation.
        
#     Parameters
#     ----------
#     X : ndarray, shape (nparts, 4)
#         The transverse beam coordinate array.
#     env_params : array-like, shape (8,)
#         The beam envelope parameters.
#     moments : bool
#         If True, plot the ellipse defined by the second-order moments of the
#         distribution.
#     limits : (umax, upmax)
#         The maximum position (umax) and slope (upmax) in the plot windows. All
#         plot windows are made square, so xmax = ymax = -xmin = -ymax = umax, and
#         and xpmax = ypmax = -xpmin = -ypmin = upmax. If None, auto-ranging is
#         performed. Alternatively, a list of tuples can be passed: [(umin, umax),
#         (upmin, upmax)]. If either element is None, auto-ranging is performed
#         on those coordinates.
#     zero_center : bool
#         If true, center the plot window on the origin. Otherwise, center the
#         plot window on the projected means of the distribution.
#     samples : int
#         The number of randomly sampled points to use in the scatter plots. If
#         None, use all the points.
#     pad : float
#         Padding for the axis ranges: umax_new = (1 + pad) * 0.5 * w, where w is
#         the width of the distribution (max - min).
#     figsize : tuple or int
#         Size of the figure (x_size, y_size). If an int is provided, the number
#         is used as the size for both dimensions.
#     dims : str or tuple
#         If 'all', plot all 6 phase space projections. Otherwise provide a tuple
#         like ('x', 'yp') which plots x vs. y'.
#     kind : {'scatter', 'scatter_density', 'hist', 'kde'}
#         The kind of plot to make on the off-diagonal subplots. Note: the 'kde'
#         and 'hist' options are not implemented yet.
#     diag_kind : {'hist', 'kde', 'none'}
#         The kind of plot to make on the diagonal subplots. If 'none', these are
#         excluded and a 3x3 grid is produced.
#     hist_height : float in range [0, 1]
#         Reduce the height of the histograms on the diagonal, which normally
#         extend to the top of the plotting window, by this factor.
#     units : str or bool
#         Whether to display units on the axis labels. Options are 'mm-mrad' or
#         'm-rad'. No units are displayed if None.
#     norm_labels : bool
#         Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
#     text : str
#         If provided, annotate the figure with `text`.
#     ax : matplotlib.pyplot.Axes object
#         If plotting only a 2D projection of the data (for example if
#         dims=('x','y'), the data will be plotted on this axis.)
#     {plt, diag, env, text}_kws : dict
#         Key word arguments. They are passed to the following functions:
#         * plt_kws  : `plt.scatter`. This doesn't need to be passed as a dict.
#                      For example, `s=10` can be added to the function call to
#                      change the marker size.
#         * diag_kws : `plt.hist`. More options will be added in the future like
#                      kde.
#         * env_kws  : `plt.plot`. For plotting the envelope ellipses.
#         * text_kws : `plt.annotate`. For any text displayed on the figure.
        
#     Returns
#     -------
#     numpy.ndarray
#         The array of subplots.
#     """
#     plt_diag = diag_kind not in ('none', None)
    
#     # Set default key word arguments
#     diag_kws = dict() if diag_kws is None else diag_kws
#     env_kws = dict() if env_kws is None else env_kws
#     text_kws = dict() if text_kws is None else text_kws
#     if kind == 'hist':
#         samples = None
#         plt_kws.setdefault('bins', 50)
#     if kind == 'scatter' or kind == 'scatter_density':
#         plt_kws.setdefault('s', 3)
#         plt_kws.setdefault('c', 'steelblue')
#         plt_kws.setdefault('marker', '.')
#         plt_kws.setdefault('ec', 'none')
#         plt_kws.setdefault('zorder', 5)
#         diag_kws.setdefault('color', plt_kws['c'])
#     if kind == 'scatter_density':
#         plt_kws.pop('c', None)
#     if diag_kind == 'hist':
#         diag_kws.setdefault('histtype', 'step')
#         diag_kws.setdefault('bins', 'auto')
#         diag_kws.setdefault('color', 'k')
#     elif diag_kind == 'kde':
#         diag_kws.setdefault('lw', 1)
#     env_kws.setdefault('color', 'k')
#     env_kws.setdefault('lw', 1)
#     env_kws.setdefault('zorder', 6)
#     text_kws.setdefault('horizontalalignment', 'center')
    
#     # Get data
#     if samples is None:
#         X_samp = X
#     else:
#         X_samp = utils.rand_rows(X, samples)
#     X_env = None
#     if env_params is not None:
#         X_env = get_ellipse_coords(env_params, npts=100)

#     # Determine axis limits.
#     if limits is None:
#         limits = auto_limits(X, pad, zero_center)
#     if len(limits) == 2:
#         limits = 2 * limits
        
#     # Create figure
#     fig, axes = setup_corner(
#         limits, figsize, norm_labels, units, dims=dims, plt_diag=plt_diag,
#         label_kws={'fontsize':'medium'}
#     )
        
#     # Plot only one projection
#     if dims != 'all':
#         j, i = [var_indices[dim] for dim in dims]
#         x, y = X[:, j], X[:, i]
#         x_samp, y_samp = X_samp[:, j], X_samp[:, i]
#         ax = axes
#         if kind == 'scatter':
#             ax.scatter(x_samp, y_samp, **plt_kws)
#         elif kind == 'scatter_density':
#             ax = scatter_density(ax, x, y, samples=samples, **plt_kws)
#         elif kind == 'hist':
#             ax.hist2d(x, y, **plt_kws)
#         if X_env is not None:
#             x_env, y_env = X_env[:, j], X_env[:, i]
#             ax.plot(x_env, y_env, **env_kws)
#         return axes
        
#     # Diagonal plots
#     if plt_diag:
#         scatter_axes = axes[1:, :-1]
#         for i, (ax, data) in enumerate(zip(axes.diagonal(), X.T)):
#             if diag_kind == 'kde':
#                 gkde = scipy.stats.gaussian_kde(data)
#                 lim = limits[i % 2]
#                 ind = np.linspace(-lim, lim, 1000)
#                 ax.plot(ind, gkde.evaluate(ind), **diag_kws)
#             elif diag_kind == 'hist':
#                 g = ax.hist(data, **diag_kws)
#         # Change height
#         top_left_ax = axes[0, 0]
#         new_ylim = (1.0 / hist_height) * top_left_ax.get_ylim()[1]
#         top_left_ax.set_ylim(0, new_ylim)
#     else:
#         scatter_axes = axes
        
#     # Scatter plots
#     for i in range(3):
#         for j in range(i + 1):
#             ax = scatter_axes[i, j]
#             x, y = X[:, j], X[:, i + 1]
#             x_samp, y_samp = X_samp[:, j], X_samp[:, i + 1]
#             if kind == 'scatter':
#                 ax.scatter(x_samp, y_samp, **plt_kws)
#             elif kind == 'scatter_density':
#                 scatter_density(ax, x, y, samples=samples, **plt_kws)
#             elif kind == 'hist':
#                 ax.hist2d(x, y, range=(ax.get_xlim(), ax.get_ylim()), **plt_kws)
#             if X_env is not None:
#                 x_env, y_env = X_env[:, j], X_env[:, i + 1]
#                 ax.plot(x_env, y_env, **env_kws)
#     if moments:
#         rms_ellipses(4 * np.cov(X.T), axes=scatter_axes, **env_kws)
    
#     if text:
#         text_pos = (0.35, 0) if plt_diag else (0.35, 0.5)
#         axes[1, 2].annotate(text, xy=text_pos, xycoords='axes fraction',
#                             **text_kws)
#     return axes
    
    
def corner_env(
    params, fill=False, pad=0.5, space=None, figsize=None, dims='all',
    cmap=None, cmap_range=(0, 1), units='mm-mrad', norm_labels=False, ax=None,
    legend_kws=None, fill_kws={}, label_kws={}, **plt_kws
):
    """Plot the 6 transverse phase space ellipses of the beam.
    
    Inputs
    ------
    params : ndarray, shape (8,) or (n, 8)
        Envelope parameters [a, b, a', b', e, f, e', f']. If multiple rows are
        provided, each will be plotted as different ellipse.
    pad : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of the 
        plot will be at umax * (1 + pad).
    space : float
        Size of the space between subplots. If None, the `tight_layout`
        optimizer is used to determined the spacing.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions.
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ('x', 'yp') which plots x vs. y'.
    ec, fc : str
        Color of the ellipse boundary (ec) and interior (fc). If either are
        None,
    cmap : Matplotlib colormap
        If plotting a sequence of envelopes, this sets the color cycle. If
        None, it will use the default color cycle. If we provide something
        like plt.cm.viridis, the different envelopes will be perceptually
        uniform from blue to yellow.
    cmap_range : (min, max)
        The locations for the color cycle to to start and end in the color map.
        (0, 1) would use the entire color map, while (0.5, 1) would start at
        the midpoint and go until the end.
    units : str or bool
        Whether to display units on the axis labels. Options are 'mm-mrad' or
        'm-rad'. No units are displayed if None.
    norm_labels : boolean
        If True, add '_n' to the axis labels. E.g. 'x' -> 'x_n'.
    ax : matplotlib.pyplot.Axes object
        If plotting only a 2D projection of the data (for example if
        dims=('x','y'), the data will be plotted on this axis.)
    legend_kws : dict
        Key word arguments for the legend.
    label_kws : dict
        Key word arguments for the axis labels.
        
    Returns
    -------
    axes : Matplotlib axes object
        3x3 array of Axes objects.
    """    
    # Get ellipse boundary data
    if type(params) is not np.ndarray:
        params = np.array(params)
    if params.ndim == 1:
        params = params[np.newaxis, :]
    coords = [get_ellipse_coords(p, npts=100) for p in params]
    limits = (1 + pad) * max_u_up_global(coords)
    
    # Set default key word arguments
    color = None if len(params) > 1 else 'k'
    plt_kws.setdefault('lw', None)
    plt_kws.setdefault('color', color)
    plt_kws.setdefault('zorder', 10)
    fill_kws.setdefault('lw', 1)
    fill_kws.setdefault('fc', 'lightsteelblue')
    fill_kws.setdefault('ec', 'k')
    fill_kws.setdefault('zorder', 10)
            
    # Create figure
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             dims=dims, label_kws={'fontsize':'medium'})
    if dims != 'all':
        ax = axes
        
    # Set color cycle
    if len(params) > 1 and cmap is not None:
        start, end = cmap_range
        colors = [cmap(i) for i in np.linspace(start, end, len(params))]
        color_cycle = cycler('color', colors)
        if dims != 'all':
            ax.set_prop_cycle(color_cycle)
        else:
            for ax in axes.flat:
                ax.set_prop_cycle(color_cycle)
            
    # Plot data
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
    if legend_kws:
        axes[1, 1].legend(**legend_kws)
    return axes
    
    
def fft(ax, x, y):
    """Compute and plot the FFT of two signals x and y on the same figure.
    
    Uses scipy.fft package. Particularly useful for plotting the horizontal
    tunes of a particle from its turn-by-turn x and y coordinates.

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
#     ax.set_xticks(np.arange(0, 0.55, 0.05))
    return ax


def scatter_density(ax, x, y, samples=None, sort=True, bins=40,
                    method='interp', **kws):
    """Scatter plot with colors weighted by local density.
    
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


def vector(ax, v, origin=(0, 0), c='k', lw=None, style='->', head_width=0.4,
           head_length=0.8):
    """Plot 2D vector `v` as an arrow."""
    props = {}
    props['arrowstyle'] = '{},head_width={},head_length={}'.format(
        style, head_width, head_length)
    props['shrinkA'] = props['shrinkB'] = 0
    props['fc'] = props['ec'] = c
    props['lw'] = lw
    ax.annotate('', xy=(origin[0]+v[0], origin[1]+v[1]),
                xytext=origin, arrowprops=props)
    return ax


def eigvec_trajectory(ax, M, i='x', j='y', colors=('r','b'), s=None, lw=2,
                      alpha=0.3):
    """Plot the trajectory of the eigenvectors of the transfer matrix.
    
    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
        The axis on which to plot.
    M : ndarray, shape (4, 4)
        The lattice transfer matrix.
    i{j} : str or int
        The component of the eigenvector to plot on the x{y} axis. Can be
        either {'x', 'xp', 'y', 'yp'} or {0, 1, 2, 3}.
    colors : two-element list or tuple
        The colors to use for eigenvector 1 and 2.
    s : float
        The marker size for the scatter plot.
    lw : float
        The lineweight for the arrrows.
    alpha : float
        The alpha parameter for the scatter plot.
        
    Returns
    -------
    ax : matplotlib.pyplot.axes object
        The modified axis.
    """
    def track(x, M, nturns=1):
        X = [x]
        for i in range(nturns):
            X.append(np.matmul(M, X[i]))
        return np.array(X)
    
    if type(i) is str:
        i, j = var_indices[i], var_indices[j]
        
    eigvals, eigvecs = np.linalg.eig(M)
    v1, _, v2, _ = eigvecs.T
    for v, color in zip((v1, v2), colors):
        X = track(v.real, M, nturns=20)
        ax.scatter(X[:, i], X[:, j], s=s, c=color, alpha=alpha, marker='o')
        vector(ax, v[[i, j]].real, c=color, lw=lw)
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
    ax.add_patch(Ellipse(center, 2*c1, 2*c2, -np.degrees(angle), **plt_kws))
    return ax


def rms_ellipses(Sigmas, figsize=(5, 5), pad=0.5, axes=None, 
                 cmap=None, cmap_range=(0, 1), **plt_kws):
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
    
    dims = {0:'x', 1:'xp', 2:'y', 3:'yp'}
    for l, Sigma in enumerate(Sigmas):
        for i in range(3):
            for j in range(i + 1):
                angle, c1, c2 = rms_ellipse_dims(Sigma, dims[j], dims[i + 1])
                if colors is not None:
                    plt_kws['color'] = colors[l]
                ellipse(axes[i, j], c1, c2, angle, **plt_kws)
    return axes




















def pair_grid(
    n_dims, figsize=None, limits=None, space=None, spines=False,
    labels=None, label_kws=None, tick_kws=None, 
    constrained_layout=True
):
    """I realize Seaborn does this with PairGrid. I've had trouble with shared 
    axes limits between diagonal and non-diagonal subplots. I've also had trouble
    with plotting 2D histograms with a dark background. So this function is okay
    for my purposes.
    """
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



def get_bin_centers(bin_edges):
    """Get bin centers assuming evenly spaced bins."""
    return bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    
    
def corner(X, figsize=None, kind='hist', limits=None, pad=0., bins=50, hist_height_frac=0.6, 
           diag_kws=None, **plot_kws):
    
    if kind =='scatter' or kind == 'scatter_density':
        plot_kws.setdefault('s', 3)
        plot_kws.setdefault('c', 'black')
        plot_kws.setdefault('marker', '.')
        plot_kws.setdefault('ec', 'none')
        plot_kws.setdefault('zorder', 5)
    elif kind == 'hist':
        plot_kws.setdefault('cmap', 'dusk_r')
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('histtype', 'step')
    diag_kws.setdefault('bins', 'auto')
    diag_kws.setdefault('color', 'black')

    n_dims = X.shape[1]
    
    if figsize is None:
        f = n_dims * 7.5 / 6.0
        figsize = (1.025 * f, f)
    
    space = None
    spines = False
    labels = ["x [mm]", "x' [mrad]", "y [mm]", "y' [mrad]", "z [m]", "dE [MeV]"]
    label_kws = None
    tick_kws = None
    
    zero_center = False
    if limits is None:
        limits = auto_limits(X, pad, zero_center)

    fig, axes = pair_grid(n_dims, figsize=figsize, limits=limits, 
                          space=space, spines=spines, labels=labels, 
                          label_kws=label_kws, tick_kws=tick_kws)

    for i, ax in enumerate(axes.diagonal()):
        ax.hist(X[:, i], range=limits[i], **diag_kws)
        
    for i in range(1, len(axes)):
        for j in range(i):
            ax = axes[i, j]
            x, y = X[:, j], X[:, i]
            if kind == 'scatter':
                ax.scatter(x, y, **plot_kws)
            elif kind == 'hist':
                ax.hist2d(x, y, bins, (limits[j], limits[i]), **plot_kws)                
           
    # Reduce height of 1D histograms. 
    max_hist_height = 0.
    for ax in axes.diagonal():
        max_hist_height = max(max_hist_height, ax.get_ylim()[1])
    max_hist_height /= hist_height_frac
    for ax in axes.diagonal():
        ax.set_ylim(0, max_hist_height)
        
    return axes