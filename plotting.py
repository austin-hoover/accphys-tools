"""
This module provides functions to visualize distributions of particles
in phase space.
"""

# 3rd party
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
import seaborn as sns
from scipy.fft import fft
from scipy.stats import gaussian_kde
from pandas.plotting._matplotlib.tools import _set_ticks_props

# My modules
from . import envelope_analysis as ea
from .utils import add_to_dict, rand_rows

# Global variables
_labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
_labels_norm = [r"$x_n$", r"$x_n'$", r"$y_n$", r"$y_n'$"]


def get_u_up_max(X):
    """Get the maximum x{y} and x'{y'} extents for any frame in `coords`.

    X : NumPy array, shape (nparts, 4)
        The beam coordinate array.
    """
    xmax, xpmax, ymax, ypmax = 2 * np.std(X, axis=0)
    umax, upmax = max(xmax, ymax), max(xpmax, ypmax)
    return (umax, upmax)
    
    
def get_u_up_max_global(coords):
    """Get the maximum x{y} and x'{y'} extents for any frame in `coords`.

    `coords` : NumPy array, shape (nframes, nparts, 4)
        The beam coordinate arrays at each frame.
    """
    u_up_local_maxes = np.array([get_u_up_max(X) for X in coords])
    umax_global, upmax_global = np.max(u_up_local_maxes, axis=0)
    return (umax_global, upmax_global)

    
def setup_corner_axes_3x3(limits, gap=0.1, figsize=(7, 7), norm_labels=False):
    """Set up lower left corner of 4x4 grid of subplots.
    
    O O O O
    X O O O
    X X O O
    X X X O
    
    It is used to plot the 6 unique pairwise relationships between 4
    variables. For example, if our variables are a, b, c, and d, the subplots
    would contain the following variables, where the vertical variable is
    printed first:
        | b-a |
        | c-a | c-b |
        | d-a | d-b | d-c
    
    Inputs
    ------
    limits : tuple
        (`umax`, `upmax`), where u can be x or y. `umax` is the maximum extent
        of the real-space plot windows, while `upmax` is the maximum extent of
        the phase-space plot windows.
    gap : float
        Size of the gap between subplots.
    figsize : tuple
        Size of the figure (x-size, y-size)
    norm_labels : boolean
        If True, add an 'n' subscript to axis labels. E.g. 'x' -> 'x_n'

    Returns
    -------
    axes : Matplotlib axes object
        3x3 array of Axes objects.
    """
    # Create figure
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row',
                             figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    
    # Configure axis limits, ticks, and labels
    labels = _labels_n if norm_labels else _labels
    umax, upmax = limits
    limits = [(-umax, umax), (-upmax, upmax)] * 2
    utick, uptick = umax * 0.8, upmax * 0.8
    ticks = [[-utick, 0, utick], [-uptick, 0, uptick]] * 2
    ticks = np.around(ticks, decimals=1)
    xlimits, xticks, xlabels = limits[:-1], ticks[:-1], labels[:-1]
    ylimits, yticks, ylabels = limits[1:], ticks[1:], labels[1:]
    
    # Edit axes
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            [ax.spines[s].set_visible(False) for s in ('top', 'right')]
            ax.set_xticks(xticks[j])
            ax.set_yticks(yticks[i])
            ax.set_xlim(xlimits[j])
            ax.set_ylim(ylimits[i])
            if i < j:
                ax.axis('off')
    for i in range(3):
        axes[i, 0].set_ylabel(ylabels[i], fontsize='xx-large')
        axes[2, i].set_xlabel(xlabels[j], fontsize='xx-large')
    
    return fig, axes
    
    
def corner(
    X,
    env_params=None,
    mm_mrad=False,
    samples=2000,
    limits=None,
    padding=0.5, 
    figsize=(7, 7),
    gap=0.1,
    s=10,
    c='tab:blue',
    hist=False,
    hist_height=0.7,
    plt_kws={},
    hist_kws={},
    figname=None,
    dpi=300,
    norm_labels=False
):
    """Plot lower corner of scatter matrix of distribution, envelope, or both.
    
    A random sample of the particles are plotted in the scatter plots while
    keeping all particles for the histograms. The dimensions will be the
    same for x and y (also x' and y').
        
    Parameters
    ----------
    X : NumPy array, shape (nparts, 4)
        The beam coordinate array.
    env_params : array-like
        The envelope parameters [a, b, a', b', e, f, e', f'].
    mm_mrad : boolean
        If True, convert coordinates from m-rad to mm-mrad.
    samples : int
        Number of randomly sampled data points to use in scatter plots.
        Default: 2,500.
    limits : tuple
        Manually set the maximum rms position and slope of the distribution 
        (umax, upmax). If None, auto-ranging is performed.
    padding : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of
        the plot will be at umax * (1 + padding).
    figsize : tuple, 
        The x and y size of the figure.
    gap : float
        Width of the gap between the subplots.
    s : float
        Marker size.
    c : str
        Marker color.
    hist : bool
        Whether to plot histograms on the diagonals.
    hist_height : float
        Reduce the height of the histogram from its default value. Must be in
        range [0, 1]. The new height will be hist_height * old_height.
    plt_kws : dict
        Keyword arguments for matplotlib.pyplot.scatter.
    hist_kws : dict
        Keyword arguments for matplotlib.pyplot.hist.
    figname : str
        Name of saved figure. 
    dpi : int
        Dots per inch of saved figure.
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
        
    Returns
    -------
    axes : Matplotlib axes object
        4x4 array of Axes objects.
    """
    # Add user supplied keyword arguments
    add_to_dict(plt_kws, 's', s)
    add_to_dict(plt_kws, 'c', c)
    add_to_dict(plt_kws, 'marker', '.')
    add_to_dict(plt_kws, 'edgecolors', 'none')
    add_to_dict(hist_kws, 'histtype', 'step')
    add_to_dict(hist_kws, 'bins', 'auto')
    add_to_dict(hist_kws, 'color', c if type(c) is str else None)
    
    # Get envelope data
    X_env = None
    if env_params is not None:
        env_params = np.array(env_params)
        X_env = ea.get_coords(env_params)
        
    # Convert from m-rad to mm-mrad
    if mm_mrad:
        coords = 1000 * np.copy(coords)
        if env_params is not None:
            env_data *= 1000
    
    # Setup figure
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    
    # Take random sample
    X_samp = rand_rows(X, samples)
    
    # Plot data
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            if i > j: 
                ax.scatter(X_samp[:, j], X_samp[:, i], **plt_kws)
                if env_params is not None:
                    ax.plot(X_env[:, j], X_env[:, i], 'k-')
            elif i == j:
                ax.hist(X[:, i], **hist_kws)
    
    # Configure axis limits
    if limits is None: 
        umax, upmax = get_u_up_max(X)
    else:
        umax, upmax = limits
    umax, upmax = (1 + padding) * np.array([umax, upmax])
    limits = 2 * [(-umax, umax), (-upmax, upmax)]
    
    if not hist:
        plt.close()
        return corner_nohist(X_samp, X_env, (umax, upmax), padding,
                             figsize, gap, figname, dpi, norm_labels,
                             **plt_kws)
    
    # Edit axes
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            # set visibility of axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < j:
                ax.axis('off')
            elif i == j:
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                ax.set_ylabel('')
            # set limits
            ax.set_xlim(limits[j])
            if i != j:
                ax.set_ylim(limits[i])
            # set ticks and ticklabels
            if i == j:
                ax.set_yticks([])
            if i != 3:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            # set labels
            if j == 0 and i != 0:
                ax.set_ylabel(_labels[i], fontsize='x-large')
            else:
                ax.set_ylabel('')
            if i == 3:
                ax.set_xlabel(_labels[j], fontsize='x-large')
            else:
                ax.set_xlabel('')
                
    # Edit histogram height
    for i in range(4):
        ax = axes[i, i]
        new_ylim = (1.0 / hist_height) * ax.get_ylim()[1]
        ax.set_ylim(0, new_ylim)
            
    # Final edits
    fig.align_labels()
    fig.set_tight_layout(True)
        
    # Save figure
    if figname is not None:
        plt.savefig(figname, dpi=dpi)
    return axes
    
    
def corner_nohist(
    X,
    X_env=None,
    limits=(1, 1),
    padding=0.25,
    figsize=(7, 7),
    gap=0.1,
    figname=None,
    dpi=300,
    norm_labels=False,
    **plt_kws
):
    """Same as `corner` but without histograms on the diagonal. Do not call
    directly... use `corner` with hist=False."""
    fig, axes = setup_corner_axes_3x3(limits, gap, figsize, norm_labels)
    hdata, vdata = X[:, :-1], X[:, 1:]
    if X_env is not None:
        hdata_env, vdata_env = X_env[:, :-1], X_env[:, 1:]
    for i in range(3):
        for j in range(3):
            if i >= j:
                ax = axes[i, j]
                ax.scatter(hdata[:, j], vdata[:, i], **plt_kws)
                if X_env is not None:
                    ax.plot(hdata_env[:, j], vdata_env[:, i], 'k-')
    # Final edits
    fig.align_labels()
    fig.set_tight_layout(True)

    # Save figure
    if figname is not None:
        plt.savefig(figname, dpi=dpi)
    return axes


def corner_env(
    params,
    mm_mrad=False,
    limits=None,
    padding=0.5,
    gap=0.1,
    figsize=(6,6),
    edgecolor='black',
    facecolor=None,
    cmap=None,
    labelsize=8,
    norm_labels=False,
    legend_kws=None,
    tight_layout=False,
    figname=None,
    dpi=None,
):
    """Plot the 6 transverse phase space ellipses of the beam.
    
    Inputs
    ------
    params : array-like
        Envelope parameters [a, b, a', b', e, f, e', f']. If a list of
        these vectors is provided, each one will be plotted.
    init_params : array-like
        Initial envelope parameters (will be plotted in the background).
    mm_mrad : boolean
        Converts coordinates from m-rad to mm-mrad.
    limits : tuple
        Manually set the maximum rms position and slope of the distribution 
        (umax, upmax). If None, auto-ranging is performed.
    padding : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of the 
        plot will be at umax * (1 + padding).
    gap : float
        Width of the gap between the subplots.
    figsize : tuple
        The (x, y) size of the figure.
    edgecolor : str
        The color of the ellipse boundary.
    facecolor : str
        The color of the ellipse interior. If None, do not fill.
    cmap : Matplotlib colormap
        If plotting a sequence of envelopes, this sets the color cycle. If
        None, it will plot distinct or random colors. If we provide something
        like plt.cm.viridis, the different envelopes will be perceptually
        uniform from blue to yellow.
    labelsize : float or str
        The size of the tick labels.
    norm_labels : boolean
        If True, add '_n' to the axis labels. E.g. 'x' -> 'x_n'.
    legend_kws : dict
        Key word args for the legend.
    tight_layout : bool
        Whether to call fig.set_tight_layout(True)
    figname : str
        Name of saved figure -> plt.savefig(figname, dpi=dpi).
    dpi : int
        Dots per inch of saved figure -> plt.savefig(figname, dpi=dpi).
        
    Returns
    -------
    axes : Matplotlib axes object
        3x3 array of Axes objects.
        
    To do
    -----
    Allow `cmap` to be a string.
    """    
    # Get ellipse boundary data for x, x', y, y'
    def get_ellipse_data(pvec):
        data = ea.get_coords(pvec).T
        if mm_mrad:
            data *= 1000
        return data
    
    if type(params) != list:
        params = [params]
    
    coords = np.array([get_ellipse_data(pvec).T for pvec in params])
    limits = get_u_up_max_global(coords)
                
    # Set up figure
    fig, axes = setup_corner_axes_3x3(limits, gap, figsize, norm_labels)
    if len(params) > 1 and cmap is not None:
        colorcycle = [cmap(i) for i in np.linspace(0, 1, len(params))]
        for ax in axes.flatten():
            ax.set_prop_cycle('color', colorcycle)

    # Plot data
    for X in coords:
        X_horiz, X_vert = X[:, :-1], X[:, 1:]
        for i in range(3):
            for j in range(3):
                ax = axes[i,j]
                if i >= j:
                    if facecolor is not None:
                        ax.fill(X_horiz[:, j], X_vert[:, i],
                                facecolor=facecolor, edgecolor='k', lw=1)
                    else:
                        color = edgecolor if len(params) == 1 else None
                        ax.plot(X_horiz[:, j], X_vert[:, i], color=color, lw=2)
    # Add legend
    if legend_kws is not None:
        axes[1, 1].legend(**legend_kws)
    
    _set_ticks_props(axes, xlabelsize=labelsize, xrot=0,
                     ylabelsize=labelsize, yrot=0)
    fig.align_labels()
    fig.set_tight_layout(tight_layout)
        
    # Save
    if figname is not None:
        plt.savefig(figname, dpi=dpi)
    return axes
    

def plot_fft(x, y, grid=True, figname=None):
    """Compute and plot the FFT of two signals x and y on the same figure.
    
    Uses scipy.fft package. Particularly useful for plotting the horizontal
    tunes of a particle from its turn-by-turn x and y coordinates.

    Let N be the number of samples, M=N/2, x(t)=signal as function of time, 
    and w(f) be the FFT of x. Then w[0] is zero frequency component, w[1:M] 
    are positive frequency components, and w[M:N] are negative frequency 
    components.
    
    Inputs
    ------
    x{y} : Numpy array, shape (n,): 
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
    xf = (1/M) * abs(fft(x)[:M])
    yf = (1/M) * abs(fft(y)[:M])

    fig, ax = plt.subplots()
    ax.set_xlabel('Tune')
    ax.set_ylabel('Amplitude')
    ax.plot(f[1:], xf[1:],label=r'$\nu_x$')
    ax.plot(f[1:], yf[1:], label=r'$\nu_y$')
    ax.legend(**legend_kws)
    ax.set_xticks(np.arange(0, 0.55, 0.05))
    ax.grid(grid)
    if figname is not None:
        plt.savefig(figname, dpi=300)
    return ax


def scatter_color_by_density(x, y, ax=None):
    """Scatter plot with color weighted by density.
    
    Taken from StackOverflow answer by Joe Kington: "https://stackoverflow.com
    /questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-
    matplotlib"
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=4, edgecolor='')
    return ax
