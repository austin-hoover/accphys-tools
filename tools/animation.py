"""
This module contains functions to animate the evolution of a beam of
particles in phase space.

To do:
    * Add option to save animation.
    * Add more plotting options to `corner`, such as kde.
"""

# Standard
from cycler import cycler
# Third party
import numpy as np
import matplotlib
import pandas as pd
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker
from matplotlib import animation
from pandas.plotting._matplotlib.tools import _set_ticks_props
from matplotlib.patches import Ellipse, transforms
# Local
from .envelope_analysis import get_ellipse_coords
from .plotting import setup_corner, get_u_up_max_global
from .utils import merge_dicts

# Settings
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# Module level variables
plt_kws_default = dict(ms=2, color='steelblue', marker='.', zorder=5, lw=0,
                       markeredgewidth=0, fillstyle='full')
diag_kws_default = dict(histtype='step', bins='auto', color='steelblue')
env_kws_default = dict(color='k', lw=1, zorder=6)
text_kws_default = dict()
artists_list = [] # Python can't find the variable, even when I declare it as
                  # global within the function. Can't figure out why.
                  

def corner(
    coords, env_params=None, samples=2000, skip=0, pad=0.5, space=0.15,
    figsize=(7, 7), kind='scatter', diag_kind='hist', hist_height=0.7,
    units='mm-mrad', norm_labels=False, text_fmt='', text_vals=None, fps=1,
    diag_kws={}, env_kws={}, text_kws={}, **plt_kws
):
    """Frame-by-frame phase space projections of the beam.
    
    Parameters
    ----------
    coords : list or ndarray, shape (nframes, nparts, 4)
        Each element contains the transverse beam coordinate array at a
        particular frame.
    env_params : list or ndarray, shape (nframes, 8) [optional]
        The envelope parameters at each frame. They are not plotted if none
        are provided.
    samples : int
        The number of randomly sampled particles to use in the off-diagonal
        subplots.
    skip : int
        The coordinates will be plotted every skip + 1 frames.
    pad : float
        Padding for the axis ranges. The edge of the plots will be at
        umax * (1 + pad), where umax is maximum amplitude of any beam particle
        among all the frames.
    space : float
        Width of the space between the subplots.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions.
    kind : {'scatter', 'hist', 'kde'}
        The kind of plot to make on the off-diagonal subplots. Note: the 'kde'
        and 'hist' options are not implemented yet.
    diag_kind : {'hist', 'kde', 'none'}
        The kind of plot to make on the diagonal subplots. If 'none', these are
        excluded and a 3x3 grid is produced. Note: the 'kde' option is not
        implemented yet.
    hist_height : float in range [0, 1]
        Reduce the height of the histograms on the diagonal, which normally
        extend to the top of the plotting window, by this factor.
    units : str or bool
        Whether to display units on the axis labels. Options are 'mm-mrad' or
        'm-rad'. No units are displayed if None.
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
    text_vals, text_fmt: list, str
        Each new frame will display text indicating the turn, position, etc..
        For example: 'Turn = 5' or 's = 17.4 m'. The string that will be printed
        is `text_fmt.format(text_vals[f])`, where f is the frame number. If
        `text_vals` is None, we use list(range(nframes)).
    fps : int
        Frames per second.
    {plt, diag, env, text}_kws : dict
        Key word arguments. They are passed to the following functions:
        * plt_kws  : `plt.plot`. For the scatter plots. This doesn't need to be
                     passed as a dict; for example, `ms=10` can be added to the
                     function call to change the marker size.
        * diag_kws : `plt.hist`. For the histograms on the diagonal. More
                     options will be added in the future like kde.
        * env_kws  : `plt.plot`. For plotting the envelope ellipses.
        * text_kws : `plt.annotate`. For any text displayed on the figure.
        
    Returns
    -------
    anim : output from matplotlib.animation.FuncAnimation
    """
    plt_env = env_params is not None
    
    # Configure key word arguments
    plt_kws = merge_dicts(plt_kws_default, plt_kws)
    if diag_kws_default['color'] != plt_kws['color']:
        diag_kws_default['color'] = plt_kws['color']
    diag_kws = merge_dicts(diag_kws_default, diag_kws)
    env_kws = merge_dicts(env_kws_default, env_kws)
    text_kws = merge_dicts(text_kws_default, text_kws)
            
    # Process particle coordinates
    if type(coords) is list:
        coords = np.array(coords)
    nframes = coords.shape[0]
    if len(coords.shape) == 2: # single particle bunch
        coords = coords[:, np.newaxis, :]
    coords_env = get_ellipse_coords(env_params) if plt_env else None
        
    # Configure text updates
    if text_vals is None:
        text_vals = list(range(nframes))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = [text_fmt.format(val) for val in text_vals]
        
    # Skip frames
    plot_every = skip + 1
    coords = coords[::plot_every]
    if plt_env:
        coords_env = coords_env[::plot_every]
    texts = texts[::plot_every]
    nframes = coords.shape[0]
    
    # Take random sample of particles for scatter plots
    coords_samp, (nframes, nparts, ndim) = coords, coords.shape
    if nparts > samples:
        idx = np.random.choice(nparts, samples, replace=False)
        coords_samp = coords[:, idx, :]
    
    # Create figure
    plt_diag = diag_kind != 'none'
    limits = (1 + pad) * get_u_up_max_global(coords)
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             plt_diag, fontsize='small')
    _set_ticks_props(axes, xlabelsize='small', ylabelsize='small')
    plt.close()

    if not plt_diag:
        return _corner_nodiag(fig, axes, coords_samp, coords_env, texts, fps,
                              env_kws, text_kws, **plt_kws)

    # Compute the maximum histogram height to keep the ylimit fixed
    max_heights = np.zeros((nframes, 4))
    bins = diag_kws['bins']
    for i, X in enumerate(coords):
        for j in range(4):
            max_heights[i, j] = np.max(np.histogram(X[:, j], bins=bins)[0])
    axes[0, 0].set_ylim(0, np.max(max_heights) / hist_height)

    # Create array of Line2D objects
    lines = [[], [], [], []]
    lines_env = [[], [], [], []]
    for i in range(4):
        for j in range(i):
            line, = axes[i, j].plot([], [], **plt_kws)
            lines[i].append(line)
            if plt_env:
                line, = axes[i, j].plot([], [], **env_kws)
                lines_env[i].append(line)

    def init():
        """Plot the background of each frame."""
        for i in range(4):
            for j in range(i):
                lines[i][j].set_data([], [])
                if plt_env:
                    lines_env[i][j].set_data([], [])
                    
    def update(t):
        """Animation function to be called sequentially."""
        X, X_samp = coords[t], coords_samp[t]
        for i in range(4):
            for j in range(i):
                lines[i][j].set_data(X_samp[:, j], X_samp[:, i])
                if plt_env:
                    X_env = coords_env[t]
                    lines_env[i][j].set_data(X_env[:, j], X_env[:, i])
        # Remove old histograms
        global artists_list
        for artists in artists_list:
            for artist in artists:
                artist.remove()
        # Plot new histograms
        artists_list = []
        for i, ax in enumerate(axes.diagonal()):
            heights, bin_edges, artists = ax.hist(X[:, i], **diag_kws)
            artists_list.append(artists)
        # Display text
        for old_text in axes[1, 2].texts:
            old_text.set_visible(False)
        axes[1, 2].annotate(texts[t], xy=(0.35, 0), xycoords='axes fraction',
                            **text_kws)

    # Call animator and possibly save the animation
    anim = animation.FuncAnimation(fig, update, frames=nframes, blit=False,
                                   interval=1000/fps)
    return anim
    
    
def _corner_nodiag(fig, axes, coords, coords_env, texts, fps, env_kws,
                   text_kws, **plt_kws):
    """Corner plot without diagonal. Helper function for `corner` method."""
    plt_env = coords is not None
    nframes = coords.shape[0]
    lines = [[], [], []]
    lines_env = [[], [], []]
    for i in range(3):
        for j in range(i + 1):
            line, = axes[i, j].plot([], [], **plt_kws)
            lines[i].append(line)
            if plt_env:
                line, = axes[i, j].plot([], [], **env_kws)
                lines_env[i].append(line)
    def init():
        """Plot the background of each frame."""
        for i in range(3):
            for j in range(i):
                lines[i][j].set_data([], [])
                if plt_env:
                    lines_env[i][j].set_data([], [])
    def update(t):
        """Animation function to be called sequentially."""
        X = coords[t]
        hdata, vdata = X[:, :-1], X[:, 1:]
        if plt_env:
            X_env = coords_env[t]
            hdata_env, vdata_env = X_env[:, :-1], X_env[:, 1:]
        for i in range(3):
            for j in range(i + 1):
                lines[i][j].set_data(hdata[:, j], vdata[:, i])
                if plt_env:
                    lines_env[i][j].set_data(hdata_env[:, j], vdata_env[:, i])
                    
        for old_text in axes[1, 2].texts:
            old_text.set_visible(False)
        axes[1, 2].annotate(texts[t], xy=(0.35, 0.5), xycoords='axes fraction',
                            **text_kws)
                            
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes,
                                   interval=1000/fps)
    return anim
    

def corner_env(
    params, skip=0, figsize=(5, 5), pad=0.25, space=0.15, ec='k',
    fc='lightsteelblue', fill=True, plot_boundary=True, show_init=False,
    clear_history=True, text_fmt='', text_vals=None, units='mm-mrad',
    norm_labels=False, fps=5, cmap=None
):
    """Corner plot with beam envelope (ellipse) only.

    Inputs
    ------
    params : ndarray, shape (nframes, 8)
        If shape is (nframes, 8), gives the envelope parameters at each frame.
        If a list of these arrays is provided, each envelope in the list will
        be plotted.
    skip : int
        The coordinates will be plotted every skip + 1 frames.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions.
    pad : float
        Padding for the axis ranges. The edge of the plots will be at
        umax * (1 + pad), where umax is maximum amplitude of any beam particle
        among all the frames.
    space : float
        Size of the space between subplots.
    ec : str
        Color of ellipse boundary.
    fc : str
        Color of the ellipse interior.
    fill : bool
        Whether to fill the ellipse.
    plot_boundary : bool
        Whether to plot the ellipse boudary.
    show_init : bool
        Whether to show static initial envelope in background.
    clear_history : bool
        Whether to clear the previous frames before plotting the new frame.
    text_vals, text_fmt: list, str
        Each new frame will display text indicating the turn, position, etc..
        For example: 'Turn = 5' or 's = 17.4 m'. The string that will be printed
        is `text_fmt.format(text_vals[f])`, where f is the frame number. If
        `text_vals` is None, we use list(range(nframes)).
    units : str or bool
        Whether to display units on the axis labels. Options are 'mm-mrad' or
        'm-rad'. No units are displayed if None.
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
    fps : int
        Frames per second.
    cmap : str
        The colormap to use as a cycler if plotting multiple envelopes. If
        None, use matplotlib's default cycler.

    Returns
    -------
    anim : output from matplotlib.animation.FuncAnimation
    """
    # Get ellipse coordinates
    if params.ndim == 2:
        params = params[np.newaxis, :]
    n_envelopes, nframes, _ = params.shape
    coords_list = [get_ellipse_coords(p) for p in params]
    if n_envelopes > 1:
        fill = False
        ec = None
    
    # Configure text updates
    if text_vals is None:
        text_vals = list(range(nframes))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = [text_fmt.format(val) for val in text_vals]
        
    # Skip frames
    for i in range(n_envelopes):
        coords_list[i] = coords_list[i][::skip+1]
    texts = texts[::skip+1]
    nframes = coords_list[0].shape[0]
        
    # Store initial ellipse
    X_init = coords_list[0][0]
    hdata_init, vdata_init = X_init[:, :-1], X_init[:, 1:]
        
    # Configure axis limits
    limits_list = np.array([(1 + pad) * get_u_up_max_global(coords)
                            for coords in coords_list])
    limits = np.max(limits_list, axis=0)

    # Create figure
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             plt_diag=False, fontsize='small')
    _set_ticks_props(axes, xlabelsize='small', ylabelsize='small')
    if cmap is not None:
        colors = [cmap(i) for i in np.linspace(0, 1, n_envelopes)]
        for ax in axes.flat:
            ax.set_prop_cycle(cycler('color', colors))
    plt.close()
    
    # Create list of Line2D objects
    lines_list = []
    for coords in coords_list:
        lines = [[], [], []]
        for i in range(3):
            for j in range(i + 1):
                ax = axes[i, j]
                line, = ax.plot([], [], '-', lw=1, color=ec)
                lines[i].append(line)
        lines_list.append(lines)

    def update(t):
        """Animation function to be called sequentially."""
        if clear_history:
            for ax in axes.flat:
                for patch in ax.patches:
                    patch.remove()
                    
        for i, coords in enumerate(coords_list):
            X = coords[t]
            hdata, vdata = X[:, :-1], X[:, 1:]
            for j in range(3):
                for k in range(j + 1):
                    ax = axes[j, k]
                    
                    if plot_boundary:
                        lines_list[i][j][k].set_data(hdata[:, k], vdata[:, j])
                    if fill:
                        ax.fill(hdata[:, k], vdata[:, j], fc=fc, lw=0)
                    if show_init and clear_history:
                        ax.plot(hdata_init[:, k], vdata_init[:, j], 'k--',
                                lw=0.5, alpha=0.25)
        # Display text
        for old_text in axes[1, 2].texts:
            old_text.set_visible(False)
        axes[1, 2].annotate(texts[t], xy=(0.35, 0.5), xycoords='axes fraction')
                        
    # Call animator and (maybe) save the animation
    anim = animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=1000/fps)
    return anim
