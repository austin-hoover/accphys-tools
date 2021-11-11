"""Various animation functions

TO DO
* Redo `corner`.
"""
from cycler import cycler
import copy
from tqdm import trange, tqdm

import numpy as np
import matplotlib
import pandas as pd
import seaborn as sns
import scipy
from matplotlib import pyplot as plt, ticker
from matplotlib import animation
from matplotlib.patches import Ellipse, transforms
from skimage import filters

from .beam_analysis import get_ellipse_coords
# from .plotting import setup_corner
from .plotting import pair_grid
from .plotting import max_u_up, max_u_up_global
from .plotting import auto_limits
from .plotting import process_limits
from .plotting import auto_n_bins_4D
from .plotting import remove_annotations
from .plotting import vector
from .plotting import var_indices
from .plotting import get_bin_centers
from .utils import rand_rows


plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    
# I had to make this global for some reason... don't remember why.
artists_list = []


def skip_frames(frames, skip=1, keep_last=False):
    last = frames[-1]
    frames = frames[::skip+1]
    if keep_last:
        if type(frames) is list:
            frames.append(last)
        elif type(frames) is np.ndarray:
            frames = np.append(frames, last)
    return frames


# def corner(
#     coords, env_params=None, limits=None, zero_center=True, dims='all', samples=None, 
#     skip=0, keep_last=False, pad=0.5, space=0.15, figsize=None, kind='scatter',
#     diag_kind='hist', hist_height=0.6, units='mm-mrad', norm_labels=False,
#     text_fmt='', text_vals=None, fps=1, diag_kws={}, env_kws={}, text_kws={},
#     **plot_kws
# ):
#     """Frame-by-frame phase space projections of the beam.

#     Parameters
#     ----------
#     coords : list or ndarray, shape (n_frames, nparts, 4)
#         Each element contains the transverse beam coordinate array at a
#         particular frame. Each frame can have a different number of particles.
#     env_params : ndarray, shape (n_frames, 8)
#         The envelope parameters at each frame. They are not plotted if none
#         are provided.
#     limits : (umax, upmax)
#         Maximum position and angle for plot windows.
#     dims : str or tuple
#         If 'all', plot all 6 phase space projections. Otherwise provide a tuple
#         like ('x', 'yp') which plots x vs. y'.
#     samples : int
#         The number of randomly sampled particles to use in the off-diagonal
#         subplots.
#     skip : int
#         The coordinates will be plotted every skip + 1 frames.
#     keep_last : bool
#         Whether to keep the last frame if skipping frames. Could probably
#         encompass this in the 'skip' parameter somehow.
#     pad : float
#         Padding for the axis ranges. The edge of the plots will be at
#         umax * (1 + pad), where umax is maximum amplitude of any beam particle
#         among all the frames.
#     space : float
#         Width of the space between the subplots.
#     figsize : tuple or int
#         Size of the figure (x_size, y_size). If an int is provided, the number
#         is used as the size for both dimensions. Default (6, 6) with diagonals
#         (5, 5) without diagonals, or (3, 3) if only one subplot.
#     kind : {'scatter', 'hist', 'kde'}
#         The kind of plot to make on the off-diagonal subplots. Note: the 'kde'
#         and 'hist' options are not implemented yet.
#     diag_kind : {'hist', 'kde', 'none'}
#         The kind of plot to make on the diagonal subplots. If 'none', these are
#         excluded and a 3x3 grid is produced. Note: the 'kde' option currently
#         does not work.
#     hist_height : float in range [0, 1]
#         Reduce the height of the histograms on the diagonal, which normally
#         extend to the top of the plotting window, by this factor.
#     units : str or bool
#         Whether to display units on the axis labels. Options are 'mm-mrad' or
#         'm-rad'. No units are displayed if None.
#     norm_labels : bool
#         Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
#     text_vals, text_fmt: list, str
#         Each new frame will display text indicating the turn, position, etc..
#         For example: 'Turn = 5' or 's = 17.4 m'. The string that will be printed
#         is `text_fmt.format(text_vals[f])`, where f is the frame number. If
#         `text_vals` is None, we use list(range(n_frames)).
#     fps : int
#         Frames per second.
#     {plt, diag, env, text}_kws : dict
#         Key word arguments. They are passed to the following functions:
#         * plot_kws  : `plt.plot`. For the scatter plots. This doesn't need to be
#                      passed as a dict; for example, `ms=10` can be added to the
#                      function call to change the marker size.
#         * diag_kws : `plt.hist`. For the histograms on the diagonal. More
#                      options will be added in the future like kde.
#         * env_kws  : `plt.plot`. For plotting the envelope ellipses.
#         * text_kws : `plt.annotate`. For any text displayed on the figure.

#     Returns
#     -------
#     matplotlib.animation.FuncAnimation
#     """
#     plt_env = env_params is not None
#     plt_diag = diag_kind not in ['none', None]

#     # Set default key word arguments
#     if kind == 'scatter':
#         if 's' in plot_kws:
#             ms = plot_kws['s']
#             plot_kws.pop('s', None)
#             plot_kws['ms'] = ms
#         if 'c' in plot_kws:
#             color = plot_kws['c']
#             plot_kws.pop('c', None)
#             plot_kws['color'] = color
#         plot_kws.setdefault('ms', 2)
#         plot_kws.setdefault('color', 'steelblue')
#         plot_kws.setdefault('marker', '.')
#         plot_kws.setdefault('zorder', 5)
#         plot_kws.setdefault('lw', 0)
#         plot_kws.setdefault('markeredgewidth', 0)
#         plot_kws.setdefault('fillstyle', 'full')
#         diag_kws.setdefault('color', plot_kws['color'])
#     elif kind == 'hist':
#         plot_kws.setdefault('cmap', 'viridis')
#         plot_kws.setdefault('bins', 50)
#     if diag_kind == 'hist':
#         diag_kws.setdefault('histtype', 'step')
#         diag_kws.setdefault('bins', 'auto')
#     elif diag_kind == 'kde':
#         diag_kws.setdefault('lw', 1)
#     env_kws.setdefault('color', 'k')
#     env_kws.setdefault('lw', 1)
#     env_kws.setdefault('zorder', 6)

#     # Process particle coordinates
#     nparts_list = np.array([X.shape[0] for X in coords])
#     if type(coords) is np.ndarray:
#         coords = [X for X in coords]
#     n_frames = len(coords)    
#     if plt_env:
#         coords_env = np.array([get_ellipse_coords(p) for p in env_params])
#     else:
#         coords_env = None

#     # Configure text updates
#     if text_vals is None:
#         text_vals = list(range(n_frames))
#     if text_fmt is None: # display empty text
#         text_fmt = ''
#     texts = np.array([text_fmt.format(val) for val in text_vals])
    
#     # Skip frames
#     coords = skip_frames(coords, skip, keep_last)
#     if plt_env:
#         coords_env = skip_frames(coords_env, skip, keep_last)
#     texts = skip_frames(texts, skip, keep_last)
#     n_frames = len(coords)
        
#     # Take random sample of particles for scatter plots
#     nparts_is_static = all([n == nparts_list[0] for n in nparts_list])
#     if nparts_is_static:
#         nparts = nparts_list[0]
#         coords_samp = np.array(coords)
#         if samples and samples < nparts:
#             idx = np.random.choice(nparts, samples, replace=False)
#             coords_samp = coords[:, idx, :]
#     else:
#         # Note that this section takes a different sample of particles on
#         # each frame.
#         coords_samp = coords
#         for i, X in enumerate(coords_samp):
#             if samples and X.shape[0] > samples:                
#                 coords_samp[i] = rand_rows(X, samples)
            
#     # Axis limits. Please clean up this section.   
#     mins_list = np.vstack([np.min(X, axis=0) for X in coords])
#     maxs_list = np.vstack([np.max(X, axis=0) for X in coords])
#     mins = np.min(mins_list, axis=0)[:4]
#     maxs = np.max(maxs_list, axis=0)[:4]
#     means = 0.5 * (mins + maxs)
#     widths = (1 + pad) * np.abs(maxs - mins)

#     width_u = max(widths[0], widths[2])
#     width_up = max(widths[1], widths[3])
#     xmin = means[0] - 0.5 * width_u
#     xmax = means[0] + 0.5 * width_u
#     ymin = means[2] - 0.5 * width_u
#     ymax = means[2] + 0.5 * width_u
#     xpmin = means[1] - 0.5 * width_up
#     xpmax = means[1] + 0.5 * width_up
#     ypmin = means[3] - 0.5 * width_up
#     ypmax = means[3] + 0.5 * width_up
    
#     if zero_center:
#         xmax = max(abs(xmin), abs(xmax))
#         ymax = max(abs(ymin), abs(ymax))
#         xpmax = max(abs(xpmin), abs(xpmax))
#         ypmax = max(abs(ypmin), abs(ypmax))
#         xmin = -xmax
#         ymin = -ymax
#         xmin = -xmax
#         ypmin = -ypmax
            
#     # Determine axis limits.
#     if limits is None:
#         limits = [(xmin, xmax), (xpmin, xpmax), (ymin, ymax), (ypmin, ypmax)]
#     if len(limits) == 2:
#         limits = 2 * limits
        
#     # Create figure
#     fig, axes = setup_corner(
#         limits, figsize, norm_labels, units, space, plt_diag, dims=dims,
#         label_kws={'fontsize':'medium'}
#     )    
#     plt.close()
    
#     if dims != 'all':
#         return _corner_2D(fig, axes, coords_samp, coords_env, dims, texts, fps,
#                           env_kws, text_kws, **plot_kws)

#     # Create array of Line2D objects.
#     lines = [[], [], []]
#     lines_env = [[], [], []]
#     lines_onepart = [[], [], []]
#     joint_axes = axes[1:, :-1] if plt_diag else axes
#     for i in range(3):
#         for j in range(i + 1):
#             if kind == 'scatter':
#                 line, = joint_axes[i, j].plot([], [], **plot_kws)
#                 lines[i].append(line)
#             if plt_env:
#                 line, = joint_axes[i, j].plot([], [], **env_kws)
#                 lines_env[i].append(line)
                
#             line, = joint_axes[i, j].plot([], [], marker='.', ms=4, c='red', zorder=999)
#             lines_onepart[i].append(line) 

#     # Compute maximum 1D histogram height among all frames and 1D projections.
#     if plt_diag:
#         max_heights = []
#         for X in coords:
#             for j in range(4):
#                 heights, bin_edges = np.histogram(X[:, j], 'auto', limits[j])
#                 max_heights.append(np.max(heights))
#         axes[0, 0].set_ylim(0, np.max(max_heights) / hist_height)
        
#     # Compute maximum 2D histogram height among all frames and 2D projections.
#     max_heights = np.zeros((3, 3))
#     if kind == 'hist':
#         for X in coords:
#             for i in range(3):
#                 for j in range(i + 1):
#                     H, xedges, yedges = np.histogram2d(X[:, j], X[:, i + 1], plot_kws['bins'],
#                                                       [limits[j], limits[i + 1]])
#                     max_heights[i, j] = max(np.max(H), max_heights[i, j])
                    
#     def init():
#         """Plot the background of each frame."""
#         for i in range(3):
#             for j in range(i):
#                 lines[i][j].set_data([], [])
#                 if plt_env:
#                     lines_env[i][j].set_data([], [])  
    
#     def update(t):
#         """Animation function to be called sequentially."""
#         remove_annotations(axes)
#         X, X_samp = coords[t], coords_samp[t]
#         for i in range(3):
#             for j in range(i + 1):
# #                 lines_onepart[i][j].set_data(X[0, j], X[0, i+1])
#                 if kind == 'scatter':
#                     lines[i][j].set_data(X_samp[:, j], X_samp[:, i+1])
#                 elif kind == 'hist':
#                     brange = (limits[j], limits[i + 1])
#                     bins = plot_kws['bins']
#                     plot_kws_temp = copy.deepcopy(plot_kws)
#                     del plot_kws_temp['bins']
#                     joint_axes[i, j].hist2d(X_samp[:, j], X_samp[:, i+1], 
#                                             bins, brange, vmax=max_heights[i, j],
#                                             **plot_kws_temp)
#                 if plt_env:
#                     X_env = coords_env[t]
#                     lines_env[i][j].set_data(X_env[:, j], X_env[:, i+1])
#         if plt_diag:
#             if diag_kind == 'hist':
#                 global artists_list
#                 for artists in artists_list:
#                     for artist in artists:
#                         artist.remove()
#                 artists_list = []
#                 for i, ax in enumerate(axes.diagonal()):
#                     heights, bin_edges, artists = ax.hist(X[:, i], **diag_kws)
#                     artists_list.append(artists)
#             elif diag_kind == 'kde':
#                 for i, ax in enumerate(axes.diagonal()):
#                     for line in ax.lines:
#                         ax.remove()
#                     gkde = scipy.stats.gaussian_kde(X[:, i])
#                     umax = limits[i % 2]
#                     ind = np.linspace(-umax, umax, 1000)
#                     ax.plot(ind, gkde.evaluate(ind), **diag_kws)

#         # Display text
#         location = (0.35, 0) if plt_diag else (0.35, 0.5)
#         axes[1, 2].annotate(texts[t], xy=location, xycoords='axes fraction', 
#                             **text_kws)

#     # Call animator and possibly save the animation
#     anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False,
#                                    interval=1000/fps)
#     return anim


def _corner_2D(fig, ax, coords, coords_env, dims, texts, fps, env_kws,
               text_kws, **plot_kws):
    """2D scatter plot (helper function for `corner`)."""
    n_frames = coords.shape[0]
    j, i = [var_indices[dim] for dim in dims]
    coords_x, coords_y = coords[:, :, j], coords[:, :, i]
    
    line, = ax.plot([], [], **plot_kws)
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
        ax.set_title(texts[t], **text_kws)
        
    return animation.FuncAnimation(fig, update, init_func=init, frames=n_frames,
                                   interval=1000/fps)
    

def corner_env(
    params, dims='all', skip=0, keep_last=False, figsize=None, grid=True,
    pad=0.25, space=0.15, ec='k', fc='lightsteelblue', lw=1, fill=True,
    plot_boundary=True, show_init=False, clear_history=True, text_fmt='',
    text_vals=None, units='mm-mrad', norm_labels=False, fps=1 , cmap=None,
    cmap_range=(0, 1), figname=None, dpi=None, bitrate=-1
):
    """Corner plot with beam envelope (ellipse) only.

    Parameters
    ----------
    params : ndarray, shape (n_frames, 8)
        If shape is (n_frames, 8), gives the envelope parameters at each frame.
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
        `text_vals` is None, we use list(range(n_frames)).
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
    matplotlib.animation.FuncAnimation
    """
    # Get ellipse coordinates
    params_list = np.copy(params)
    if params_list.ndim == 2:
        params_list = params_list[np.newaxis, :]
    n_envelopes, n_frames, _ = params_list.shape
    coords_list = []
    for params in params_list:
        coords = np.array([get_ellipse_coords(p) for p in params])
        coords_list.append(coords)
    X_init = coords_list[0][0]
    if n_envelopes > 1:
        fill = False
        ec = None
            
    # Configure text updates
    if text_vals is None:
        text_vals = list(range(n_frames))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = np.array([text_fmt.format(val) for val in text_vals])
    
    # Skip frames
    for i in range(n_envelopes):
        coords_list[i] = skip_frames(coords_list[i], skip, keep_last)
    texts = skip_frames(texts, skip ,keep_last)
    n_frames = coords_list[0].shape[0]
        
    # Configure axis limits
    limits_list = np.array([(1 + pad) * max_u_up_global(coords)
                            for coords in coords_list])
    limits = np.max(limits_list, axis=0)

    # Create figure
    fig, axes = setup_corner(limits, figsize, norm_labels, units, space,
                             dims=dims, plt_diag=False)
                             
    if cmap is not None:
        values = np.linspace(cmap_range[0], cmap_range[1], len(coords_list))
        colors = [cmap(i) for i in values]
        if dims != 'all':
            ax = axes
            ax.set_prop_cycle(cycler('color', colors))
        else:
            for ax in axes.flat:
                ax.set_prop_cycle(cycler('color', colors))
                             
    if not grid:
        if dims != 'all':
            ax = axes
            ax.grid(False)
        else:
            for ax in axes.flat:
                ax.grid(False)
                
    plt.close()
                
    if dims != 'all':
        return _corner_env_2D(fig, ax, coords_list, dims, clear_history,
                              show_init, plot_boundary, fill, fc, ec, lw,
                              fps, texts)
    
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
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000/fps)
    if figname:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(figname, writer=writer, dpi=dpi)
    return anim
    
    
def _corner_env_2D(fig, ax, coords_list, dims, clear_history, show_init,
                   plot_boundary, fill, fc, ec, lw, fps, texts):
    X_init = coords_list[0][0]
    lines = []
    for coords in coords_list:
        line, = ax.plot([], [], '-', lw=lw, color=ec)
        lines.append(line)
    k, j = [var_indices[dim] for dim in dims]
    
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
            ax.set_title(texts[t])
    return animation.FuncAnimation(fig, update, frames=coords_list[0].shape[0],
                                   interval=1000/fps)
        

def corner_onepart(
    X, dims='all', vecs=None, show_history=False, skip=0, pad=0.35, space=0.15,
    figsize=None, units='mm-mrad', norm_labels=False, text_fmt='', limits=None, zero_center=True,
    text_vals=None, fps=1, figname=None, dpi=None, bitrate=-1, text_kws={},
    label_kws={}, tick_kws={}, tickm_kws={}, grid_kws={}, history_kws={},
    **plot_kws
):
    # Set default key word arguments
    for kws in (plot_kws, history_kws):
        if 's' in kws:
            ms = kws['s']
            kws.pop('s', None)
            kws['ms'] = ms
        if 'c' in kws:
            color = kws['c']
            kws.pop('c', None)
            kws['color'] = color
        kws.setdefault('ms', 8)
        kws.setdefault('color', 'k')
        kws.setdefault('marker', '.')
        kws.setdefault('zorder', 5)
        kws.setdefault('lw', 0)
        kws.setdefault('markeredgewidth', 0)
        kws.setdefault('fillstyle', 'full')
    history_kws.setdefault('zorder', 0)
    
    # Configure text updates
    n_frames = X.shape[0]
    if text_vals is None:
        text_vals = list(range(n_frames))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = [text_fmt.format(val) for val in text_vals]
        
    # Skip frames
    X = X[::skip+1]
    if vecs is not None:
        vecs = np.array(vecs)
        vecs = vecs[:, ::skip+1]
    texts = texts[::skip+1]
    n_frames = X.shape[0]
    
    # Determine axis limits.
    if limits is None:
        limits = auto_limits(X, pad, zero_center)
    if len(limits) == 2:
        limits = 2 * limits
    
    # Create figure
    fig, axes = setup_corner(
        limits, figsize, norm_labels, units, space, plt_diag=False, dims=dims,
        tick_kws=tick_kws, tickm_kws=tickm_kws, label_kws=label_kws
    )
    ax_list = [axes] if dims != 'all' else axes.flat
    if grid_kws:
        for ax in ax_list:
            ax.grid(**grid_kws)
    plt.close()

    # Create Line2D objects
    if dims != 'all':
        line, = axes.plot([], [], **plot_kws)
        line_history, = axes.plot([], [], **history_kws)
    else:
        lines_history = [[], [], []]
        lines = [[], [], []]
        for i in range(3):
            for j in range(i + 1):
                line, = axes[i, j].plot([], [], **plot_kws)
                lines[i].append(line)
                line, = axes[i, j].plot([], [], **history_kws)
                lines_history[i].append(line)

    def init():
        """Plot the background of each frame."""
        if dims != 'all':
            line.set_data([], [])
            line_history.set_data([], [])
        else:
            for i in range(3):
                for j in range(i):
                    lines[i][j].set_data([], [])
                    lines_history[i][j].set_data([], [])

    def update(t):
        """Animation function to be called sequentially."""
        remove_annotations(axes)
        _X, _Xold = X[[t]], X[:t]
        if dims != 'all':
            j, i = [var_indices[dim] for dim in dims]
            line.set_data(_X[:, j], _X[:, i])
            if show_history:
                line_history.set_data(_Xold[:, j], _Xold[:, i])
            axes.set_title(texts[t], **text_kws)
        else:
            for i in range(3):
                for j in range(i + 1):
                    ax = axes[i, j]
                    lines[i][j].set_data(_X[:, j], _X[:, i+1])
                    if show_history:
                        lines_history[i][j].set_data(_Xold[:, j], _Xold[:, i+1])
                    if vecs is not None:
                        v1, v2 = vecs[0][t], vecs[1][t]
                        vector(ax, v1[[j, i+1]], origin=(0, 0), c='r',
                                  head_width=0.2, head_length=0.4)
                        vector(ax, v2[[j, i+1]], origin=(v1[[j, i+1]]),
                                  c='b', head_width=0.2, head_length=0.4)
            axes[1, 2].annotate(texts[t], xy=(0.35, 0.5),
                                xycoords='axes fraction', **text_kws)

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=n_frames,
                                   interval=1000/fps)
    if figname:
        writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
        anim.save(figname, writer=writer, dpi=dpi)
    return anim











def auto_limits_global(coords, pad=0., zero_center=False, sigma=None):
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


def corner(
    coords, dims=None, 
    kind='hist',
    limits=None, pad=0., zero_center=False, space=0.1, figsize=None, 
    skip=0, keep_last=False, 
    hist_height_frac=1.0, 
    sigma=None,
    blur=None, 
    global_cmap_norm=True,
    static_n_bins=True,
    text_fmt='', text_vals=None, fps=1, 
    diag_kws=None, text_kws=None, 
    **plot_kws
):     
    # Set default key word arguments
    if kind == 'hist':
        plot_kws.setdefault('cmap', 'mono_r')
        plot_kws.setdefault('bins', 'auto')
        plot_kws.setdefault('shading', 'auto')
    elif kind == 'scatter':
        plot_kws.setdefault('marker', 'o')
        plot_kws.setdefault('lw', 0.)
        plot_kws.setdefault('markeredgecolor', 'none')
        plot_kws.setdefault('color', 'black')
        plot_kws.setdefault('ms', 0.3)
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    diag_kws.setdefault('histtype', 'step')
    diag_kws.setdefault('bins', 'auto')
    if text_kws is None:
        text_kws = dict()

    # Process particle coordinates
    if dims is None:
        n_dims = coords[0].shape[1]
    else:
        n_dims = dims
    n_parts_list = np.array([X.shape[0] for X in coords])
    coords = [X[:, :n_dims] for X in coords]
    n_frames = len(coords)    

    # Configure text updates
    if text_vals is None:
        text_vals = list(range(n_frames))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = np.array([text_fmt.format(val) for val in text_vals])
    
    # Skip frames
    coords = skip_frames(coords, skip, keep_last)
    texts = skip_frames(texts, skip, keep_last)
    n_frames = len(coords)
                    
    # Create figure
    if figsize is None:
        f = n_dims * 7.5 / 6.0
        figsize = (1.05 * f, f)
    space = None
    spines = False
    labels = ["x [mm]", "x' [mrad]", "y [mm]", "y' [mrad]", "z [m]", "dE [MeV]"]
    label_kws = None
    tick_kws = None
    if limits is None:
        limits = auto_limits_global(coords, pad, zero_center, sigma)
    fig, axes = pair_grid(n_dims, figsize=figsize, limits=limits, 
                          space=space, spines=spines, labels=labels, 
                          label_kws=label_kws, tick_kws=tick_kws)
    plt.close()
        
    # Compute 1D projections at each frame.
    n_frames = len(coords)    
    heights_list_1D = [[] for _ in range(n_frames)]
    n_bins_list_1D = [[] for _ in range(n_frames)]
    for frame in trange(n_frames):
        X = coords[frame]
        for i in range(n_dims):
            heights, edges = np.histogram(X[:, i], diag_kws['bins'], limits[i])
            heights_list_1D[frame].append(heights)
            n_bins_list_1D[frame].append(len(edges) - 1)
    diag_kws.pop('bins') 
    
    # Keep ylim the same across frames for all 1D projections.
    max_height = 0
    for frame in range(n_frames):
        for i in range(n_dims):
            max_height = max(max_height, np.max(heights_list_1D[frame][i]))
    axes[0, 0].set_ylim(0, max_height / hist_height_frac)
    
    # Option to keep the number of bins fixed in 2D histograms.
    n_bins_list_1D = np.array(n_bins_list_1D)
    if static_n_bins is not None:
        for j in range(n_dims):
            if static_n_bins == 'mean':
                n_bins_list_1D[:, j] = np.max(n_bins_list_1D[:, j])
            elif static_n_bins == 'max':
                n_bins_list_1D[:, j] = np.mean(n_bins_list_1D[:, j])
            elif static_n_bins == 'final':
                n_bins_list_1D[:, j] = n_bins_list_1D[-1, j]
            elif type(static_n_bins) in [int, float]:
                n_bins_list_1D[:, j] = static_n_bins * np.max(n_bins_list_1D[:, j])
    
    # Setup for 2D plots
    if kind == 'hist':  
        _bins = plot_kws.pop('bins')
        # Compute 2D projections at each frame.
        heights_list = [[[] for j in range(n_dims)] for i in range(n_dims)]
        xcenters_list = [[[] for j in range(n_dims)] for i in range(n_dims)]
        ycenters_list = [[[] for j in range(n_dims)] for i in range(n_dims)]       
        for frame in trange(n_frames):
            X = coords[frame]
            for i in range(n_dims):
                for j in range(n_dims):
                    if _bins == 'auto':
                        bins = (n_bins_list_1D[frame, j], n_bins_list_1D[frame, i])
                    else:
                        bins = _bins
                    heights, xedges, yedges = np.histogram2d(X[:, j], X[:, i], bins, (limits[j], limits[i]))
                    heights_list[i][j].append(heights)
                    xcenters_list[i][j].append(get_bin_centers(xedges))
                    ycenters_list[i][j].append(get_bin_centers(yedges))                    
        # Compute global max height for each 2D projection.
        max_heights = np.zeros((n_dims, n_dims))
        for frame in range(n_frames):
            for i in range(1, n_dims):
                for j in range(i + 1):
                    Z = heights_list[i][j][frame]
                    max_heights[i, j] = max(np.max(Z), max_heights[i, j])   
    elif kind == 'scatter':
        lines = [[] for _ in range(n_dims)]
        for i in range(n_dims):
            for j in range(i):
                line, = axes[i, j].plot([], [], **plot_kws)
                lines[i].append(line)  

    def update(frame):
        global artists_list
        for artists in artists_list:
            for artist in artists:
                artist.remove()
        artists_list = []
        remove_annotations(axes)  
        
        X = coords[frame]
                
        # Diagonal plots
        for i, ax in enumerate(axes.diagonal()):
            xmin, xmax = limits[i]
            y = heights_list_1D[frame][i]
            n = len(y)
            x = np.linspace(limits[i][0], limits[i][1], n)
            _, _, artists = ax.hist(x, n, weights=heights_list_1D[frame][i], **diag_kws)
#             heights, edges, artists = ax.hist(X[:, i], n_bins_list_1D[frame][i], **diag_kws)
            artists_list.append(artists)
            
        # Off-diagonal plots
        if kind == 'hist':
            artists = []
            for i in range(1, n_dims):
                for j in range(i):
                    ax = axes[i, j]
                    x = xcenters_list[i][j][frame]
                    y = ycenters_list[i][j][frame]
                    Z = heights_list[i][j][frame]
                    if blur:
                        Z = filters.gaussian(Z, sigma=blur)
                    if global_cmap_norm:
                        plot_kws['vmax'] = max_heights[i, j]
                    qmesh = ax.pcolormesh(x, y, Z.T, **plot_kws)
                    artists.append(qmesh)
            artists_list.append(artists)
        elif kind == 'scatter':
            for i in range(1, n_dims):
                for j in range(i):
                    lines[i][j].set_data(X[:, j], X[:, i])

        # Display text
        axes[1, 2].annotate(texts[frame], xy=(0.35, 0), xycoords='axes fraction', **text_kws)
        
    # Call animator
    interval = 1000. / fps
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False,
                                   interval=interval)
    return anim