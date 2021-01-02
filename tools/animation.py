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
import seaborn as sns
import scipy
from matplotlib import pyplot as plt, ticker
from matplotlib import animation
from pandas.plotting._matplotlib.tools import _set_ticks_props
from matplotlib.patches import Ellipse, transforms
# Local
from .envelope_analysis import get_ellipse_coords
from .plotting import setup_corner, get_u_up_max_global, remove_annotations
from .plotting import vector as arrowplot
from .utils import merge_dicts

# Settings
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# Module level variables
artists_list = []
                  

def corner(
    coords, env_params=None, vectors=None, show_history=False, samples=2000,
    skip=0, pad=0.5,
    space=0.15, figsize=(7, 7), kind='scatter', diag_kind='hist',
    hist_height=0.7, units='mm-mrad', norm_labels=False, text_fmt='',
    text_vals=None, fps=1, diag_kws={}, env_kws={}, text_kws={},
    **plt_kws
):
    """Frame-by-frame phase space projections of the beam.
    
    Parameters
    ----------
    coords : list or ndarray, shape (nframes, nparts, 4)
        Each element contains the transverse beam coordinate array at a
        particular frame.
    env_params : ndarray, shape (nframes, 8)
        The envelope parameters at each frame. They are not plotted if none
        are provided.
    vectors : list or ndarray, shape (2, nframes, 4)
        If provided, plot the eigenvectors at each frame. The first vector will
        be plotted as an arrow with its tail at the origin. The second vector
        will be plotted with its tail at the tip of the first eigenvector.
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
        excluded and a 3x3 grid is produced. Note: the 'kde' option currently
        does not work.
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
    plt_vec = vectors is not None
    
    # Set default key word arguments
    if 's' in plt_kws:
        ms = plt_kws['s']
        plt_kws.pop('s', None)
        plt_kws['ms'] = ms
    plt_kws.setdefault('ms', 2)
    plt_kws.setdefault('color', 'steelblue')
    plt_kws.setdefault('marker', '.')
    plt_kws.setdefault('zorder', 5)
    plt_kws.setdefault('lw', 0)
    plt_kws.setdefault('markeredgewidth', 0)
    plt_kws.setdefault('fillstyle', 'full')
    if diag_kind == 'hist':
        diag_kws.setdefault('histtype', 'step')
        diag_kws.setdefault('bins', 'auto')
        diag_kws.setdefault('color', plt_kws['color'])
    elif diag_kind == 'kde':
        diag_kws.setdefault('lw', 1)
    env_kws.setdefault('color', 'k')
    env_kws.setdefault('lw', 1)
    env_kws.setdefault('zorder', 6)
            
    # Process particle coordinates
    if type(coords) is list:
        coords = np.array(coords)
    nframes = coords.shape[0]
    if len(coords.shape) == 2: # single particle bunch
        coords = coords[:, np.newaxis, :]
    coords_env = get_ellipse_coords(env_params) if plt_env else None
    if plt_vec and type(vectors) in [list, tuple]:
        vectors = np.array(vectors)
        
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
    if plt_vec:
        vectors = vectors[:, ::plot_every]
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
        return _corner_nodiag(fig, axes, coords, coords_env, vectors,
                              show_history, texts, fps, env_kws, text_kws,
                              **plt_kws)

    # Compute the maximum histogram height among frames to keep the ylimit fixed
    max_heights = np.zeros((nframes, 4))
    for i, X in enumerate(coords):
        for j in range(4):
            max_heights[i, j] = np.max(np.histogram(X[:, j], bins='auto')[0])
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
        remove_annotations(axes)
        X = coords[t]
        if show_history and t > 0:
            _coords_samp = coords_samp[:t+1]
        else:
            _coords_samp = coords_samp[[t]]
        # Scatter plots
        for i in range(4):
            for j in range(i):
                for X_samp in _coords_samp:
                    lines[i][j].set_data(X_samp[:, j], X_samp[:, i])
                if plt_env:
                    X_env = coords_env[t]
                    lines_env[i][j].set_data(X_env[:, j], X_env[:, i])
                if plt_vec:
                    v1, v2 = vectors[0, t], vectors[1, t]
                    arrowplot(axes[i, j], v1[[j,i]], origin=(0, 0), c='r')
                    arrowplot(axes[i, j], v2[[j,i]], origin=(v1[[j,i]]), c='b')
        # Diagonal plots
        if diag_kind == 'hist':
            global artists_list
            for artists in artists_list:
                for artist in artists:
                    artist.remove()
            artists_list = []
            for i, ax in enumerate(axes.diagonal()):
                heights, bin_edges, artists = ax.hist(X[:, i], **diag_kws)
                artists_list.append(artists)
        elif diag_kind == 'kde':
            for i, ax in enumerate(axes.diagonal()):
                for line in ax.lines:
                    ax.remove()
                gkde = scipy.stats.gaussian_kde(X[:, i])
                umax = limits[i % 2]
                ind = np.linspace(-umax, umax, 1000)
                ax.plot(ind, gkde.evaluate(ind), **diag_kws)
                
        # Display text
        axes[1, 2].annotate(texts[t], xy=(0.35, 0), xycoords='axes fraction',
                            **text_kws)

    # Call animator and possibly save the animation
    anim = animation.FuncAnimation(fig, update, frames=nframes, blit=False,
                                   interval=1000/fps)
    return anim
    
    
def _corner_nodiag(fig, axes, coords, coords_env, vectors, show_history, texts,
                   fps, env_kws, text_kws, **plt_kws):
    """Corner plot without diagonal. Helper function for `corner` method."""
    plt_env = coords_env is not None
    plt_vec = vectors is not None
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
        remove_annotations(axes)
        if plt_env:
            X_env = coords_env[t]
        if show_history and t > 0:
            _coords = coords[:t+1]
        else:
            _coords = coords[[t]]
        for i in range(3):
            for j in range(i + 1):
                ax = axes[i, j]
                for X in _coords:
                    lines[i][j].set_data(X[:, j], X[:, i + 1])
                if plt_env:
                    lines_env[i][j].set_data(X_env[:, j], X_env[:, i + 1])
                if plt_vec:
                    v1, v2 = vectors[0, t], vectors[1, t]
                    arrowplot(ax, v1[[j, i+1]], origin=(0, 0), c='r',
                              head_width=0.2, head_length=0.4)
                    arrowplot(ax, v2[[j, i+1]], origin=(v1[[j, i+1]]),
                              c='b', head_width=0.2, head_length=0.4)
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





#-------------------------------------------------------------------------------

def corner_eigvecs(
    v1_tracked, v2_tracked, skip=0, pad=0.5, space=0.15,
    figsize=(6, 6), units='mm-mrad', norm_labels=False, text_fmt='',
    text_vals=None, fps=1, text_kws={}, **plt_kws
):
    # Configure text updates
    nframes = v1_tracked.shape[0]
    if text_vals is None:
        text_vals = list(range(nframes))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = [text_fmt.format(val) for val in text_vals]
        
    # Skip frames
    plot_every = skip + 1
    v1_tracked = v1_tracked[::plot_every]
    v2_tracked = v2_tracked[::plot_every]
    texts = texts[::plot_every]
    nframes = v1_tracked.shape[0]
    
    # Create figure
    umax1, upmax1 = (1 + pad) * get_u_up_max_global(v1_tracked)
    umax2, upmax2 = (1 + pad) * get_u_up_max_global(v1_tracked)
    umax = max(umax1, umax2)
    upmax = max(upmax1, upmax2)
    fig, axes = setup_corner((umax, upmax), figsize, norm_labels, units, space,
                              plt_diag=False, fontsize='small')
    _set_ticks_props(axes, xlabelsize='small', ylabelsize='small')
    plt.close()

    lines1 = [[], [], []]
    lines2 = [[], [], []]
    lines_list = [lines1, lines2]
    for i in range(3):
        for j in range(i + 1):
            for lines in lines_list:
                line, = axes[i, j].plot([], [], **plt_kws)
                lines[i].append(line)

    def init():
        """Plot the background of each frame."""
        for i in range(3):
            for j in range(i):
                for lines in lines_list:
                    lines[i][j].set_data([], [])

    def update(t):
        """Animation function to be called sequentially."""
        v1 = v1_tracked[t]
        v2 = v2_tracked[t]
        for i in range(3):
            for j in range(i + 1):
                lines[i][j].set_data(hdata[:, j], vdata[:, i])
                    
        for old_text in axes[1, 2].texts:
            old_text.set_visible(False)
        axes[1, 2].annotate(texts[t], xy=(0.35, 0.5), xycoords='axes fraction',
                            **text_kws)
                            
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes,
                                   interval=1000/fps)
    return anim
