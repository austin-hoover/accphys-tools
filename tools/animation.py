"""
This module contains functions to animate the evolution of a beam of
particles in phase space.

To do:
    * Add option to save animation.
    * Add more plotting options to `corner`, such as kde. It throws an error
      right now.
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
from .plotting import (
    setup_corner,
    get_labels,
    get_u_up_max,
    get_u_up_max_global,
    remove_annotations,
    vector as arrowplot,
    str_to_int)

# Settings
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['grid.color'] = '#b0b0b0'
 
# Module level variables
artists_list = []


def skip_frames(X, skip, keep_last=False):
    nframes = X.shape[0]
    idx = np.arange(0, nframes, skip + 1)
    if keep_last:
        idx = np.append(idx, nframes - 1)
    return X[idx]


def corner(
    coords, env_params=None, dims='all', samples=2000, skip=0, keep_last=False,
    pad=0.5, space=0.15, figsize=None, kind='scatter', diag_kind='hist',
    hist_height=0.7, units='mm-mrad', norm_labels=False, text_fmt='',
    text_vals=None, fps=1, diag_kws={}, env_kws={}, text_kws={}, **plt_kws
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
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ('x', 'yp') which plots x vs. y'.
    samples : int
        The number of randomly sampled particles to use in the off-diagonal
        subplots.
    skip : int
        The coordinates will be plotted every skip + 1 frames.
    keep_last : bool
        Whether to keep the last frame if skipping frames. Could probably
        encompass this in the 'skip' parameter somehow.
    pad : float
        Padding for the axis ranges. The edge of the plots will be at
        umax * (1 + pad), where umax is maximum amplitude of any beam particle
        among all the frames.
    space : float
        Width of the space between the subplots.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions. Default (6, 6) with diagonals
        (5, 5) without diagonals, or (3, 3) if only one subplot.
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
    plt_diag = diag_kind not in ['none', None]

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

    # Configure text updates
    if text_vals is None:
        text_vals = list(range(nframes))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = np.array([text_fmt.format(val) for val in text_vals])
    
    # Skip frames
    coords = skip_frames(coords, skip, keep_last)
    if plt_env:
        coords_env = skip_frames(coords_env, skip, keep_last)
    texts = skip_frames(texts, skip, keep_last)
    nframes = coords.shape[0]
    
    # Take random sample of particles for scatter plots
    coords_samp, (nframes, nparts, ndim) = coords, coords.shape
    if nparts > samples:
        idx = np.random.choice(nparts, samples, replace=False)
        coords_samp = coords[:, idx, :]
    limits = (1 + pad) * get_u_up_max_global(coords)

    # Create figure
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             plt_diag, dims=dims, fontsize='medium')
    plt.close()
    if dims != 'all':
        return _corner_2D(fig, axes, coords_samp, coords_env, dims, texts, fps,
                          env_kws, text_kws, **plt_kws)

    # Create array of Line2D objects
    lines = [[], [], []]
    lines_env = [[], [], []]
    scatter_axes = axes[1:, :-1] if plt_diag else axes
    for i in range(3):
        for j in range(i + 1):
            line, = scatter_axes[i, j].plot([], [], **plt_kws)
            lines[i].append(line)
            if plt_env:
                line, = scatter_axes[i, j].plot([], [], **env_kws)
                lines_env[i].append(line)

    # Compute the maximum histogram height among frames to keep ylim fixed
    if plt_diag:
        max_heights = np.zeros((nframes, 4))
        for i, X in enumerate(coords):
            for j in range(4):
                max_heights[i, j] = np.max(np.histogram(X[:, j], bins='auto')[0])
        axes[0, 0].set_ylim(0, np.max(max_heights) / hist_height)

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
        X, X_samp = coords[t], coords_samp[t]
        for i in range(3):
            for j in range(i + 1):
                lines[i][j].set_data(X_samp[:, j], X_samp[:, i+1])
                if plt_env:
                    X_env = coords_env[t]
                    lines_env[i][j].set_data(X_env[:, j], X_env[:, i+1])
        if plt_diag:
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
        location = (0.35, 0) if plt_diag else (0.35, 0.5)
        axes[1, 2].annotate(texts[t], xy=location, xycoords='axes fraction',
                            **text_kws)

    # Call animator and possibly save the animation
    anim = animation.FuncAnimation(fig, update, frames=nframes, blit=False,
                                   interval=1000/fps)
    return anim


def _corner_2D(fig, ax, coords, coords_env, dims, texts, fps, env_kws,
               text_kws, **plt_kws):
    """2D scatter plot (helper function for `corner`)."""
    nframes = coords.shape[0]
    j, i = [str_to_int[dim] for dim in dims]
    coords_x, coords_y = coords[:, :, j], coords[:, :, i]
    
    line, = ax.plot([], [], **plt_kws)
    line_env, = ax.plot([], [], **env_kws)
    def init():
        line.set_data([], [])
        line_env.set_data([], [])
        
    def update(t):
        X = coords[t]
        line.set_data(X[:, j], X[:, i])
        if coords_env is not None:
            X_env = coords_env[t]
            line_env.set_data(X_env[:, j], X_env[:, i])
        
    return animation.FuncAnimation(fig, update, init_func=init, frames=nframes,
                                   interval=1000/fps)
    

def corner_env(
    params, dims='all', skip=0, keep_last=False, figsize=None, grid=True,
    pad=0.25, space=0.15, ec='k', fc='lightsteelblue', lw=None, fill=True,
    plot_boundary=True, show_init=False, clear_history=True, text_fmt='',
    text_vals=None, units='mm-mrad', norm_labels=False, fps=1 , cmap=None,
    cmap_range=(0, 1), figname=None, dpi=None, bitrate=-1
):
    """Corner plot with beam envelope (ellipse) only.

    Parameters
    ----------
    params : ndarray, shape (nframes, 8)
        If shape is (nframes, 8), gives the envelope parameters at each frame.
        If a list of these arrays is provided, each envelope in the list will
        be plotted.
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ('x', 'yp') which plots x vs. y'.
    skip : int
        The coordinates will be plotted every skip + 1 frames.
    keep_last : bool
        Whether to keep the last frame if skipping frames. Could probably
        encompass this in the 'skip' parameter somehow.
    figsize : tuple or int
        Size of the figure (x_size, y_size). If an int is provided, the number
        is used as the size for both dimensions.
    grid : bool
        Whether to plot grid lines.
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
    lw : float
        Line weight of ellipse boundary.
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
    figname, dpi, bitrate : str
        Name of file name, dpi, and bitrate of the saved animation.

    Returns
    -------
    anim : output from matplotlib.animation.FuncAnimation
    """
    # Get ellipse coordinates
    if params.ndim == 2:
        params = params[np.newaxis, :]
    n_envelopes, nframes, _ = params.shape
    coords_list = [get_ellipse_coords(p) for p in params]
    X_init = coords_list[0][0]
    if n_envelopes > 1:
        fill = False
        ec = None
            
    # Configure text updates
    if text_vals is None:
        text_vals = list(range(nframes))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = np.array([text_fmt.format(val) for val in text_vals])
    
    # Skip frames
    for i in range(n_envelopes):
        coords_list[i] = skip_frames(coords_list[i], skip, keep_last)
    texts = skip_frames(texts, skip ,keep_last)
    nframes = coords_list[0].shape[0]
        
    # Configure axis limits
    limits_list = np.array([(1 + pad) * get_u_up_max_global(coords)
                            for coords in coords_list])
    limits = np.max(limits_list, axis=0)

    # Create figure
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             dims=dims, plt_diag=False, fontsize=None)
    if not grid:
        for ax in axes.flat:
            ax.grid(False)
    if cmap is not None:
        values = np.linspace(cmap_range[0], cmap_range[1], len(coords_list))
        colors = [cmap(i) for i in values]
        for ax in axes.flat:
            ax.set_prop_cycle(cycler('color', colors))
    plt.close()
    
    if dims != 'all':
        if cmap is not None:
            ax.set_prop_cycle(cycler('color', colors))
        return _corner_env_2D(fig, axes, coords_list, dims, clear_history,
                              show_init, plot_boundary, fill, fc, ec, fps)
    
    # Create list of Line2D objects
    lines_list = []
    for coords in coords_list:
        lines = [[], [], []]
        for i in range(3):
            for j in range(i + 1):
                ax = axes[i, j]
                line, = ax.plot([], [], '-', color=ec, lw=lw)
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
                        ax.plot(X_init[:, k], X_init[:, j+1], 'k--',
                                lw=0.5, alpha=0.25)
        # Display text
        remove_annotations(axes[0, 1])
        axes[0, 1].annotate(texts[t], xy=(0.35, 0.5), xycoords='axes fraction')
                        
    # Call animator and (maybe) save the animation
    anim = animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=1000/fps)
    if figname:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(figname, writer=writer, dpi=dpi)
    return anim
    
    
def _corner_env_2D(fig, ax, coords_list, dims, clear_history,
                   show_init, plot_boundary, fill, fc, ec, fps):
    X_init = coords_list[0][0]
    lines = []
    for coords in coords_list:
        line, = ax.plot([], [], '-', lw=1, color=ec)
        lines.append(line)
    k, j = [str_to_int[dim] for dim in dims]
    
    def update(t):
        if clear_history:
             for patch in ax.patches:
                 patch.remove()
        for line, coords in zip(lines, coords_list):
            X = coords[t]
            if plot_boundary:
                line.set_data(X[:, k], X[:, j])
            if fill:
                ax.fill(X[:, k], X[:, j], fc=fc, lw=0)
            if show_init and clear_history:
                ax.plot(X_init[:, k], X_init[:, j], 'k--',
                        lw=0.5, alpha=0.25)
    return animation.FuncAnimation(fig, update, frames=coords_list[0].shape[0],
                                   interval=1000/fps)
        

def corner_onepart(
    X, dims='all', vecs=None, show_history=False, skip=0, pad=0.35,
    space=0.15, figsize=None, grid=True, units='mm-mrad', norm_labels=False,
    text_fmt='', text_vals=None, fps=1, figname=None, dpi=None, bitrate=-1,
    text_kws={}, grid_kws={}, **plt_kws
):
    # Set default key word arguments
    if 's' in plt_kws:
        ms = plt_kws['s']
        plt_kws.pop('s', None)
        plt_kws['ms'] = ms
    plt_kws.setdefault('ms', 8)
    plt_kws.setdefault('color', 'k')
    plt_kws.setdefault('marker', '.')
    plt_kws.setdefault('zorder', 5)
    plt_kws.setdefault('lw', 0)
    plt_kws.setdefault('markeredgewidth', 0)
    plt_kws.setdefault('fillstyle', 'full')
    
    # Configure text updates
    nframes = X.shape[0]
    if text_vals is None:
        text_vals = list(range(nframes))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = [text_fmt.format(val) for val in text_vals]
        
    # Skip frames
    X = X[::skip+1]
    if vecs is not None:
        vecs = np.array(vecs)
        vecs = vecs[:, ::skip+1]
    texts = texts[::skip+1]
    nframes = X.shape[0]
    
    # Create figure
    limits = (1 + pad) * get_u_up_max(X)
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             plt_diag=False, dims=dims)
    if not grid:
        for ax in axes.flat:
            ax.grid(False)
    plt.close()
            
    # Create Line2D objects
    if dims != 'all':
        ax = axes
        line, = ax.plot([], [], **plt_kws)
    else:
        lines = [[], [], []]
        for i in range(3):
            for j in range(i + 1):
                line, = axes[i, j].plot([], [], **plt_kws)
                lines[i].append(line)
                
    def init():
        """Plot the background of each frame."""
        if dims != 'all':
            line.set_data([], [])
        else:
            for i in range(3):
                for j in range(i):
                    lines[i][j].set_data([], [])

    def update(t):
        """Animation function to be called sequentially."""
        _X = X[:t+1] if show_history and t > 0 else X[[t]]
        if dims != 'all':
            j, i = [str_to_int[dim] for dim in dims]
            line.set_data(_X[:, j], _X[:, i+1])
        else:
            remove_annotations(axes)
            for i in range(3):
                for j in range(i + 1):
                    ax = axes[i, j]
                    lines[i][j].set_data(_X[:, j], _X[:, i+1])
                    if vecs is not None:
                        v1, v2 = vecs[0][t], vecs[1][t]
                        arrowplot(ax, v1[[j, i+1]], origin=(0, 0), c='r',
                                  head_width=0.2, head_length=0.4)
                        arrowplot(ax, v2[[j, i+1]], origin=(v1[[j, i+1]]),
                                  c='b', head_width=0.2, head_length=0.4)
            axes[1, 2].annotate(texts[t], xy=(0.35, 0.5),
                                xycoords='axes fraction', **text_kws)
                            
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes,
                                   interval=1000/fps)
    if figname:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(figname, writer=writer, dpi=dpi)
    return anim
