"""
This module contains functions to visualize accelerator physics data.

To do:
    * Add grid option to `corner` and `corner_env`.
    * Possibly create separate file for plotting utility functions.
    * Add function: tune space resonance lines (complete; on hardrive currently)
    * Create function: tune footprint.
"""

# Standard
from cycler import cycler
# Third party
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, animation, ticker
from matplotlib.lines import Line2D
import seaborn as sns
import scipy
from pandas.plotting._matplotlib.tools import _set_ticks_props
# Local
from . import envelope_analysis as ea
from .utils import rand_rows, merge_dicts

# Module level variables
_labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
_labels_norm = [r"$x_n$", r"$x_n'$", r"$y_n$", r"$y_n'$"]
str_to_int = {'x':0, 'xp':1, 'y':2, 'yp':3}


def save(figname, dir, **kwargs):
    """Save figure to directory `dir`."""
    if not dir.endswith('/'):
        dir += '/'
    filename = ''.join([dir, figname, '.png'])
    plt.savefig(filename, facecolor='white', **kwargs)


def get_u_up_max(X):
    """Get the maximum x{y} and x'{y'} extents for any frame in `coords`.

    X : ndarray, shape (nparts, 4)
        The beam coordinate array.
    """
    xmax, xpmax, ymax, ypmax = np.max(X, axis=0)
    umax, upmax = max(xmax, ymax), max(xpmax, ypmax)
    return np.array([umax, upmax])
    
    
def get_u_up_max_global(coords):
    """Get the maximum x{y} and x'{y'} extents for any frame in `coords`.

    coords : ndarray, shape (nframes, nparts, 4)
        The beam coordinate arrays at each frame.
    """
    u_up_local_maxes = np.array([get_u_up_max(X) for X in coords])
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
    """Deleta all text (as well as arrows) from the figure."""
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
        
        
def setup_corner(
    limits=(1, 1), figsize=None, norm_labels=False, units=None, space=None,
    plt_diag=False, dims='all', **text_kws
):
    """Set up lower left corner of scatter plot matrix. A 4D example:

        X O O O
        X X O O
        X X X O
        X X X X
    
    Inputs
    ------
    limits : (umax, upmax)
        The maximum position (umax) and slope (upmax) in the plot windows.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions.
    norm_labels : boolean
        If True, add an 'n' subscript to axis labels. E.g. 'x' -> 'x_n'
    units : str
        If `m-rad` or `mm-mrad`, the appropriat units are attached to each
        axis label (such as x [mm] or x' [mrad]). Otherwise no units are
        displayed.
    space : float
        Size of the space between subplots. If None, the `tight_layout`
        optimizer is used to determined the spacing.
    plt_diag : bool
        Whether to include the subplots on the diagonal (4x4 vs. 3x3).
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ('x', 'yp') which plots x vs. y'.
    **text_kws
        Key word arguments for axis text.

    Returns
    -------
    fig, axes
    """
    # Preliminaries
    if figsize is None:
        if dims == 'all':
            figsize = 6 if plt_diag else 5
        else:
            figsize = 3
    if type(figsize) in [int, float]:
        figsize = (figsize, figsize)
    if norm_labels:
        units = None
    labels = get_labels(units, norm_labels)
    umax, upmax = limits
    limits = 2 * [(-umax, umax), (-upmax, upmax)]
    loc_u, loc_up = ticker.MaxNLocator(3), ticker.MaxNLocator(3)
    mloc_u, mloc_up = ticker.AutoMinorLocator(4), ticker.AutoMinorLocator(4)
    locators = 2 * [loc_u, loc_up]
    mlocators = 2 * [mloc_u, mloc_up]
    
    # Only 2 variables plotted
    if dims != 'all':
        fig, ax = plt.subplots(figsize=figsize)
        j, i = [str_to_int[dim] for dim in dims]
        ax.set_xlim(limits[j])
        ax.set_ylim(limits[i])
        ax.set_xlabel(labels[j])
        ax.set_ylabel(labels[i])
        ax.xaxis.set_major_locator(locators[j])
        ax.yaxis.set_major_locator(locators[i])
        ax.xaxis.set_minor_locator(mlocators[j])
        ax.yaxis.set_minor_locator(mlocators[i])
        return fig, ax
    
    nrows = ncols = 4 if plt_diag else 3
    sharey = False if plt_diag else 'row'
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex='col', sharey=sharey)
    l_col, b_row, diag = axes[:, 0], axes[-1, :], axes.diagonal()
    if space is None:
        fig.subplots_adjust(wspace=space, hspace=space)
    make_lower_triangular(axes)
    despine(axes.flat, ('top', 'right'))
    
    # Configure axis sharing
    if plt_diag:
        l_col = l_col[1:]
        for i, row in enumerate(axes[1:, :]): # share scatter plot y axis
            set_share_axes(row[:i+1], sharey=True)
        set_share_axes(diag, sharey=True) # share histogram y axis
        # Don't show y-axis on histogram subplots
        despine(diag, 'left')
        toggle_grid(diag, 'off')
    else:
        for row in axes:
            set_share_axes(row, sharey=True)

    # Set axis limits, labels, and ticks
    set_labels(b_row, labels, 'xlabel', **text_kws)
    set_labels(l_col, labels[1:], 'ylabel', **text_kws)
    fig.align_labels()
    set_limits(b_row, limits, 'x')
    set_limits(l_col, limits[1:], 'y')
    start = 0 if plt_diag else 1
    for row, loc, mloc in zip(axes, locators[start:], mlocators[start:]):
        for ax in row:
            ax.yaxis.set_major_locator(loc)
            ax.yaxis.set_minor_locator(mloc)
    for col, loc, mloc in zip(axes.T, locators, mlocators):
        for ax in col:
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_minor_locator(mloc)
    _set_ticks_props(axes, xlabelsize=None, ylabelsize=None)
    if space is None:
        plt.tight_layout(rect=[0, 0, 1.025, 0.975])
    return fig, axes

    
def corner(
    X, env_params=None, samples=2000, pad=0.5, figsize=None, dims='all',
    kind='scatter', diag_kind='hist', hist_height=0.6, units='mm-mrad',
    norm_labels=False, text=None, ax=None, diag_kws={}, env_kws={}, text_kws={},
    **plt_kws
):
    """Plot the pairwise relationships between the beam phase space coordinates.
    
    Of course other routines exist such as `scatter_matrix` in Pandas or
    `pairplot` in Seaborn; however, I could not find an option to only plot
    a small sample of the particles in the scatter plots while keeping all
    the particles for the histogram calculation.
        
    Parameters
    ----------
    X : ndarray, shape (nparts, 4)
        The transverse beam coordinate array.
    env_params : array-like, shape (8,)
        The beam envelope parameters.
    samples : int
        The number of randomly sampled points to use in the scatter plots.
    pad : float
        Padding for the axis ranges: umax_new = (1 + pad) * umax_old.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions.
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ('x', 'yp') which plots x vs. y'.
    kind : {'scatter', 'scatter_density', 'hist', 'kde'}
        The kind of plot to make on the off-diagonal subplots. Note: the 'kde'
        and 'hist' options are not implemented yet.
    diag_kind : {'hist', 'kde', 'none'}
        The kind of plot to make on the diagonal subplots. If 'none', these are
        excluded and a 3x3 grid is produced.
    hist_height : float in range [0, 1]
        Reduce the height of the histograms on the diagonal, which normally
        extend to the top of the plotting window, by this factor.
    units : str or bool
        Whether to display units on the axis labels. Options are 'mm-mrad' or
        'm-rad'. No units are displayed if None.
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
    text : str
        If provided, annotate the figure with `text`.
    ax : matplotlib.pyplot.Axes object
        If plotting only a 2D projection of the data (for example if
        dims=('x','y'), the data will be plotted on this axis.)
    {plt, diag, env, text}_kws : dict
        Key word arguments. They are passed to the following functions:
        * plt_kws  : `plt.scatter`. This doesn't need to be passed as a dict.
                     For example, `s=10` can be added to the function call to
                     change the marker size.
        * diag_kws : `plt.hist`. More options will be added in the future like
                     kde.
        * env_kws  : `plt.plot`. For plotting the envelope ellipses.
        * text_kws : `plt.annotate`. For any text displayed on the figure.
        
    Returns
    -------
    numpy.ndarray
        The array of subplots.
    """
    plt_diag = diag_kind not in ('none', None)
    
    # Set default key word arguments
    if kind == 'scatter' or kind == 'scatter_density':
        plt_kws.setdefault('s', 3)
        plt_kws.setdefault('c', 'steelblue')
        plt_kws.setdefault('marker', '.')
        plt_kws.setdefault('ec', 'none')
        plt_kws.setdefault('zorder', 5)
    if kind == 'scatter_density':
        plt_kws.pop('c', None)
    if diag_kind == 'hist':
        diag_kws.setdefault('histtype', 'step')
        diag_kws.setdefault('bins', 'auto')
    elif diag_kind == 'kde':
        diag_kws.setdefault('lw', 1)
    env_kws.setdefault('color', 'k')
    env_kws.setdefault('lw', 1)
    env_kws.setdefault('zorder', 6)
    text_kws.setdefault('horizontalalignment', 'center')
    
    # Get data
    X_samp = rand_rows(X, samples) # sample of particles for scatter plots
    X_env = None
    if env_params is not None:
        env = ea.Envelope(params=env_params)
        X_env = env.generate_dist(50, 'on_ellipse')
    limits = (1 + pad) * get_u_up_max(X) # axis limits
    
    # Create figure
    fig, axes = setup_corner(limits, figsize, norm_labels, units, dims=dims,
                             plt_diag=plt_diag, fontsize='medium')

    # Single particle
    if dims != 'all':
        j, i = [str_to_int[dim] for dim in dims]
        x, y = X_samp[:, j], X_samp[:, i]
        ax = axes
        if kind == 'scatter':
            ax.scatter(x, y, **plt_kws)
        elif kind == 'scatter_density':
            ax = scatter_density(ax, x, y, **plt_kws)
        if X_env is not None:
            ax.plot(X_env[:, j], X_env[:, i], **env_kws)
        return axes
        
    # Diagonal plots
    if plt_diag:
        scatter_axes = axes[1:, :-1]
        for ax, data, lim in zip(axes.diagonal(), X.T, limits):
            if diag_kind == 'kde':
                gkde = scipy.stats.gaussian_kde(data)
                ind = np.linspace(-lim, lim, 1000)
                ax.plot(ind, gkde.evaluate(ind), **diag_kws)
            elif diag_kind == 'hist':
                ax.hist(data, **diag_kws)
        # Change height
        top_left_ax = axes[0, 0]
        new_ylim = (1.0 / hist_height) * top_left_ax.get_ylim()[1]
        top_left_ax.set_ylim(0, new_ylim)
    else:
        scatter_axes = axes
        
    # Scatter plots
    for i in range(3):
        for j in range(i + 1):
            ax = scatter_axes[i, j]
            x, y = X_samp[:, j], X_samp[:, i+1]
            if kind == 'scatter':
                ax.scatter(x, y, **plt_kws)
            elif kind == 'scatter_density':
                scatter_density(ax, x, y, **plt_kws)
            if X_env is not None:
                ax.plot(x, y, **env_kws)
                
    if text:
        text_pos = (0.35, 0) if plt_diag else (0.35, 0.5)
        axes[1, 2].annotate(text, xy=text_pos, xycoords='axes fraction',
                            **text_kws)
    return axes
    
    
def corner_env(
    params, fill=False, pad=0.5, space=None, figsize=None, dims='all',
    cmap=None, cmap_range=(0, 1), units='mm-mrad', norm_labels=False, ax=None,
    legend_kws=None, fill_kws={}, label_kws={}, **plt_kws
):
    """Plot the 6 transverse phase space ellipses of the beam.
    
    Inputs
    ------
    params : ndarray, shape (8,) or (n, 8)
        Envelope parameters [a, b, a', b', e, f, e', f']. Each of the n rows
        will be plotted.
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
    coords = ea.get_ellipse_coords(params)
    limits = (1 + pad) * get_u_up_max_global(coords)
    
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
                             dims=dims, fontsize='medium')
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
    idx = [str_to_int[dim] for dim in dims]
    for X in coords:
        if dims != 'all':
            j, i = idx
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
    if legend_kws is not None and n > 2:
        axes[1, 1].legend(**legend_kws)
    return axes
    
    
def fft(x, y, grid=True, legend=False, figname=None):
    """Compute and plot the FFT of two signals x and y on the same figure.
    
    Uses scipy.fft package. Particularly useful for plotting the horizontal
    tunes of a particle from its turn-by-turn x and y coordinates.

    Let N be the number of samples, M=N/2, x(t)=signal as function of time, 
    and w(f) be the FFT of x. Then w[0] is zero frequency component, w[1:M] 
    are positive frequency components, and w[M:N] are negative frequency 
    components.
    
    Inputs
    ------
    x{y} : ndarray, shape (n,):
        Contains the x{y} coordinate at n time steps.
    grid : bool
        Whether to put grid on plot.
        
    Returns
    -------
    ax: Matplotlib.axes object
    """
    N = len(x)
    M = N // 2
    f = (1/N) * np.arange(M)
    xf = (1/M) * abs(scipy.fft.fft(x)[:M])
    yf = (1/M) * abs(scipy.fft.fft(y)[:M])

    fig, ax = plt.subplots()
    ax.set_xlabel('Tune')
    ax.set_ylabel('Amplitude')
    ax.plot(f[1:], xf[1:], label=r'$\nu_x$')
    ax.plot(f[1:], yf[1:], label=r'$\nu_y$')
    ax.set_xticks(np.arange(0, 0.55, 0.05))
    ax.grid(grid)
    if legend:
        ax.legend(loc='upper right', framealpha=1)
    return ax


def scatter_density(ax, x, y, **kws):
    """Scatter plot with color weighted by density.
    
    Taken from StackOverflow answer by Joe Kington: 'https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib'
    """
    # Calculate the point density
    xy = np.vstack([x, y])
    z = scipy.stats.gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
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
        i, j = str_to_int[i], str_to_int[j]
        
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
    print(eigvals)
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
