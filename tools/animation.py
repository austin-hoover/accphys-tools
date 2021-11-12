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
from .plotting import pair_grid
from .plotting import pair_grid_nodiag
from .plotting import max_u_up, max_u_up_global
from .plotting import auto_limits
from .plotting import auto_limits_global
from .plotting import process_limits
from .plotting import auto_n_bins_4D
from .plotting import remove_annotations
from .plotting import vector
from .plotting import var_indices
from .utils import get_bin_centers
from .utils import rand_rows
from . import plotting as myplt


plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    
# I had to make this global for some reason... I don't remember why. It holds
# a list of lists of artists. These can be removed or set invisible at any
# time.
ARTISTS_LIST = []


def skip_frames(frames, skip=1, keep_last=False):
    last = frames[-1]
    frames = frames[::skip+1]
    if keep_last:
        if type(frames) is list:
            frames.append(last)
        elif type(frames) is np.ndarray:
            frames = np.append(frames, last)
    return frames
    

def corner(
    coords, dims=None, kind='hist', figsize=None, limits=None, skip=0,
    keep_last=False, samples=None, hist_height_frac=0.6, text_fmt='',
    text_vals=None, blur=None, global_cmap_norm=False, static_n_bins=False,
    rms_ellipse=False, rms_ellipse_kws=None, env_params=None, env_kws=None,
    text_kws=None, grid_kws=None, autolim_kws=None, plot_kws=None,
    diag_kws=None, **anim_kws
):
    """Plot the pairwise relationships between the coordinates.
    
    The parameters are basically the same as in `animation.corner`; new
    parameters are listed below:
    
    Parameters
    ----------
    coords : list or ndarray
        List of coordinate arrays. The number of points in each array does
        not need to be the same.
    dims : int
        The number of dimensions to plot.
    skip : int
        Plot every skip + 1 frames.
    keep_last : bool
        Whether to keep the last frame if skipping frames.
    samples : int
        Random sampling is complicated by the fact that the number of points
        can change between frames. If the number of points is the same across
        all frames, the same random group of points is plotted on all frames.
        If not, a different random group is chosen on each frame.
    text_vals, text_fmt: list, str
        Each new frame will display text indicating the turn, position, etc.
        For example: 'Turn = 5' or 's = 17.4 m'. The string that will be
        printed is `text_fmt.format(text_vals[f])`, where f is the frame
        number. If `text_vals` is None, we use list(range(n_frames)).
    global_cmap_norm : bool
        If True and plotting 2D histograms: for each subplot, normalize the
        colormap using all frames for that subplot. This makes sense if raw
        counts are plotted; however, it can wash out certain frames if the
        distribution becomes very peaked in a different frame.
    env_params : list
        List of Danilov envelope parameters to be passed to `corner_env`.
    static_n_bins : {'mean', 'max', 'final', int, float}
        The auto-binning routine produces a different number of bins on each
        frame. This parameter keeps the number of bins within each subplot the
        same on all frames. There are a few ways I've thought about doing this:
            'mean': Mean number of bins across frames.
            'max': Maximum number of bins across frames.
            'min': Minimum number of bins across frames.
            'final': Number of bins at the last frame.
            int: Number of bins at frame `static_n_bins`.
            float: `static_n_bins` * maximum number of bins across frames.
    **anim_kws
        Key word arguments for matplotlib.animation.FuncAnimation.
    """
    # Set default key word arguments
    if plot_kws is None:
        plot_kws = dict()
    if kind == 'hist':
        plot_kws.setdefault('cmap', 'mono_r')
        plot_kws.setdefault('bins', 'auto')
        plot_kws.setdefault('shading', 'auto')
    elif kind == 'scatter':
        plot_kws.setdefault('ms', 3)
        plot_kws.setdefault('color', 'black')
        plot_kws.setdefault('marker', '.')
        plot_kws.setdefault('markeredgecolor', 'none')
        plot_kws.setdefault('zorder', 5)
        plot_kws.setdefault('lw', 0)
        plot_kws['lw'] = 0
        if 'c' in plot_kws:
            plot_kws['color'] = plot_kws.pop('c')
        if 's' in plot_kws:
            plot_kws['ms'] = plot_kws.pop('c')
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    diag_kws.setdefault('histtype', 'step')
    diag_kws.setdefault('bins', 'auto')
    if text_kws is None:
        text_kws = dict()
    if autolim_kws is None:
        autolim_kws = dict()
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        rms_ellipse_kws.setdefault('2rms', True)
        two_rms = rms_ellipse_kws.pop('2rms')
    if env_params is not None:
        if env_kws is None:
            env_kws = dict()

    # Process particle coordinates
    if dims is None:
        n_dims = coords[0].shape[1]
    else:
        n_dims = dims
    n_points_list = np.array([X.shape[0] for X in coords])
    coords = [X[:, :n_dims] for X in coords]
    n_frames = len(coords)

    # Configure text updates
    if text_vals is None:
        text_vals = list(range(n_frames))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = np.array([text_fmt.format(val) for val in text_vals])
    
    # Skip frames.
    coords = skip_frames(coords, skip, keep_last)
    texts = skip_frames(texts, skip, keep_last)
    if env_params is not None:
        env_params = skip_frames(env_params, skip, keep_last)
    n_frames = len(coords)
    
    # Take random sample of particles for scatter plots.
    n_points_is_static = all([n == n_points_list[0] for n in n_points_list])
    if n_points_is_static:
        coords = np.array(coords)
        coords_samp = np.copy(coords)
        n_points = n_points_list[0] # this doesn't change
        if samples and samples < n_points:
            idx = np.random.choice(n_points, samples, replace=False)
            coords_samp = coords[:, idx, :]
    else:
        coords_samp = coords
        for i, X in enumerate(coords_samp):
            if samples and X.shape[0] > samples:
                coords_samp[i] = rand_rows(X, samples)
                
    # Configure axes limits.
    if limits is None:
        limits = auto_limits_global(coords, **autolim_kws)

    # Create figure.
    if figsize is None:
        f = n_dims * 7.5 / 6.0
        figsize = (1.05 * f, f)
    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault('labels', ["x [mm]", "x' [mrad]",
                                   "y [mm]", "y' [mrad]",
                                   "z [m]", "dE [MeV]"])
    grid_kws.setdefault('limits', limits)
    grid_kws.setdefault('figsize', figsize)
    fig, axes = pair_grid(n_dims, **grid_kws)
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
    if static_n_bins:
        for j in range(n_dims):
            _n_bins_list = n_bins_list_1D[:, j]
            if static_n_bins == 'mean':
                n_bins_list_1D[:, j] = np.mean(_n_bins_list)
            elif static_n_bins == 'max':
                n_bins_list_1D[:, j] = np.max(_n_bins_list)
            elif static_n_bins == 'min':
                n_bins_list_1D[:, j] = np.min(_n_bins_list)
            elif static_n_bins == 'final':
                frame = -1
                n_bins_list_1D[:, j] = _n_bins_list[frame]
            elif type(static_n_bins) is float:
                n_bins_list_1D[:, j] = static_n_bins * np.max(_n_bins_list)
            elif type(static_n_bins) is int:
                frame = static_n_bins
                n_bins_list_1D[:, j] = _n_bins_list[frame]
            else:
                raise ValueError("Invalid `static_n_bins` parameter.")
    
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
        global ARTISTS_LIST
        for artists in ARTISTS_LIST:
            for artist in artists:
                artist.remove()
        ARTISTS_LIST = []
        remove_annotations(axes)
        
        X = coords[frame]
                
        # Diagonal plots
        for i, ax in enumerate(axes.diagonal()):
            xmin, xmax = limits[i]
            y = heights_list_1D[frame][i]
            n = len(y)
            x = np.linspace(limits[i][0], limits[i][1], n)
            _, _, artists = ax.hist(x, n, weights=heights_list_1D[frame][i], **diag_kws)
            ARTISTS_LIST.append(artists)
            
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
            ARTISTS_LIST.append(artists)
        elif kind == 'scatter':
            for i in range(1, n_dims):
                for j in range(i):
                    X = coords_samp[frame]
                    lines[i][j].set_data(X[:, j], X[:, i])
        
        # Plot ellipses.
        scatter_axes = axes[1:, :-1]
        if env_params is not None:
            _, env_lines = myplt.corner_env(
                env_params[frame], dims='all',
                axes=scatter_axes, return_lines=True,
                **env_kws
            )
            ARTISTS_LIST.append(env_lines)
        if rms_ellipse:
            Sigma = np.cov(X.T)
            if two_rms:
                Sigma *= 4.0
            _, rms_artists = myplt.rms_ellipses(
                Sigma,
                axes=scatter_axes,
                return_artists=True,
                **rms_ellipse_kws
            )
            ARTISTS_LIST.append(rms_artists)
                    
        # Display text
        axes[1, 2].annotate(texts[frame], xy=(0.1, 0),
                            xycoords='axes fraction',
                            **text_kws)
        
    anim = animation.FuncAnimation(fig, update, frames=n_frames, **anim_kws)
    return anim


def corner_env(
    params, dims='all', skip=0, keep_last=False, 
    figsize=None, limits=None, fill=True, show_init=False, clear_history=True,
    text_fmt='', text_vals=None, units='mm-mrad', norm_labels=False,
    cmap=None, cmap_range=(0, 1), autolim_kws=None, grid_kws=None,
    fill_kws=None, plt_kws=None, init_kws=None,
    **anim_kws
):
    """Corner plot with beam envelope only.

    Parameters
    ----------
    params : ndarray, shape (n_frames, 8)
        If shape is (n_frames, 8), gives the envelope parameters at each frame.
        If a list of these arrays is provided, each envelope in the list will
        be plotted.
    dims : str or tuple
        If 'all', plot all 6 phase space projections. Otherwise provide a tuple
        like ("x", "yp") or (0, 3).
    skip : int
        Plot every skip + 1 frames.
    keep_last : bool
        Whether to keep the last frame if skipping frames. 
    figsize : tuple or int
        Size of the figure (x_size, y_size). 
    limits : list
        List of (min, max) for each dimension.
    fill : bool
        Whether to fill the ellipse.
    show_init : bool
        Whether to show static initial envelope in background.
    clear_history : bool
        Whether to clear the previous frames before plotting the new frame.
    text_vals, text_fmt: list, str
        Each new frame will display text indicating the turn, position, etc.
        For example: 'Turn = 5' or 's = 17.4 m'. The string that will be
        printed is `text_fmt.format(text_vals[f])`, where f is the frame
        number. If `text_vals` is None, we use list(range(n_frames)).
    units : str, bool, or None
        Whether to display units on the axis labels. Options are 'mm-mrad' or
        'm-rad'.
    norm_labels : bool
        Whether to add '_n' to the axis labels ('x' -> 'x_n').
    cmap : list of colors, Matplotlib colormap, or str
        Determines the color cycle if plotting multiple envelopes.
    cmap_range : (min, max)
        The locations for the color cycle to start and end in the color map
        (between 0 and 1).
    autolim_kws : dict
        Key word arguments for `auto_limits_global` method.
    grid_kws : dict
        Key word arguments for `pair_grid` method.
    fill_kws : dict
        Key word arguments for `ax.fill` if filling the ellipses.
    plt_kws : dict
        Key word arguments for `ax.plot` if plotting ellipse boundaries.
    **anim_kws
        Key word arguments for matplotlib.animation.FuncAnimation.
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    # Default key word arguments.
    if fill_kws is None:
        fill_kws = dict()
    fill_kws.setdefault('lw', 0)
    fill_kws.setdefault('fc', 'lightsteelblue')
    if plt_kws is None:
        plt_kws = dict()
    plt_kws.setdefault('color', 'black')
    plt_kws.setdefault('ls', '-')
    if init_kws is None:
        init_kws = dict()
    init_kws.setdefault('color', 'black')
    init_kws.setdefault('ls', '--')
    init_kws.setdefault('lw', 0.5)
    init_kws.setdefault('alpha', 0.25)
        
    # Get ellipse coordinates at each frame.
    params_list = np.copy(params)
    if params_list.ndim == 2:
        params_list = params_list[np.newaxis, :]
    n_env, n_frames, _ = params_list.shape
    coords_list = []
    for params in params_list:
        coords = np.array([get_ellipse_coords(p) for p in params])
        coords_list.append(coords)
    X_init = coords_list[0][0]
    if n_env > 1:
        fill = False
        ec = None
            
    # Configure text updates.
    if text_vals is None:
        text_vals = list(range(n_frames))
    if text_fmt is None: # display empty text
        text_fmt = ''
    texts = [text_fmt.format(val) for val in text_vals]
    
    # Skip frames.
    for i in range(n_env):
        coords_list[i] = skip_frames(coords_list[i], skip, keep_last)
    texts = skip_frames(texts, skip, keep_last)
    n_frames = coords_list[0].shape[0]
        
    # Configure axes limits.
    if limits is None:
        if autolim_kws is None:
            autolim_kws = dict()
        autolim_kws.setdefault('pad', 0.5)
        limits = auto_limits_global(np.vstack(coords_list), **autolim_kws)
        
    # Create figure
    if grid_kws is None:
        grid_kws = dict()
    grid_kws.setdefault('figsize', figsize)
    grid_kws.setdefault('limits', limits)
    grid_kws.setdefault('constrained_layout', True)
    grid_kws.setdefault('labels', myplt.get_labels(units, norm_labels))
    if dims == 'all':
        n_dims = 4
        fig, axes = pair_grid_nodiag(n_dims, **grid_kws)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        i, j = dims
        if type(i) is str:
            i = var_indices[i]
        if type(j) is str:
            j = var_indices[j]
        ax.set_xlabel(grid_kws['labels'][i])
        ax.set_ylabel(grid_kws['labels'][j])
        ax.set_xlim(limits[i])
        ax.set_ylim(limits[j])
                                 
    if cmap is not None:
        colors = None
        if type(cmap) in [list, tuple]:
            colors = cmap
        else:
            values = np.linspace(cmap_range[0], cmap_range[1], n_env)
            colors = [cmap(i) for i in values]
        if dims != 'all':
            ax.set_prop_cycle(cycler('color', colors))
        else:
            for ax in axes.flat:
                ax.set_prop_cycle(cycler('color', colors))
    plt.close()
                
    if dims != 'all':
        return _corner_env_2D(fig, ax, coords_list, dims, fill, show_init,
                              clear_history, texts, init_kws, plt_kws,
                              fill_kws, **anim_kws)
    
    # Create list of Line2D objects.
    lines_list = []
    for coords in coords_list:
        lines = [[], [], []]
        for i in range(3):
            for j in range(i + 1):
                ax = axes[i, j]
                line, = ax.plot([], [], **plt_kws)
                lines[i].append(line)
        lines_list.append(lines)
    
    def update(t):
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
                    lines_list[i][j][k].set_data(hdata[:, k], vdata[:, j])
                    if fill:
                        ax.fill(hdata[:, k], vdata[:, j], **fill_kws)
                    if show_init and clear_history:
                        ax.plot(X_init[:, k], X_init[:, j + 1], **init_kws)
        remove_annotations(axes[0, 1])
        axes[0, 1].annotate(texts[t], xy=(0.35, 0.5), xycoords='axes fraction')

    anim = animation.FuncAnimation(fig, update, frames=n_frames, **anim_kws)
    return anim
    
    
def _corner_env_2D(fig, ax, coords_list, dims, fill, show_init, clear_history,
                   texts, init_kws, plt_kws, fill_kws, **anim_kws):
    """Helper function for `corner_env`."""
    X_init = coords_list[0][0]
    lines = []
    for coords in coords_list:
        line, = ax.plot([], [], **plt_kws)
        lines.append(line)
    k, j = [var_indices[dim] for dim in dims]
    
    def update(t):
        if clear_history:
            for patch in ax.patches:
                patch.remove()
        for line, coords in zip(lines, coords_list):
            X = coords[t]
            line.set_data(X[:, k], X[:, j])
            if fill:
                ax.fill(X[:, k], X[:, j], **fill_kws)
            if show_init and clear_history:
                ax.plot(X_init[:, k], X_init[:, j], **init_kws)
            ax.set_title(texts[t])
    frames = len(coords_list[0])
    anim = animation.FuncAnimation(fig, update, frames=frames, **anim_kws)
    return anim
        

def corner_onepart(
    X, dims='all', vecs=None, show_history=False, skip=0, pad=0.35, space=0.15,
    figsize=None, units='mm-mrad', norm_labels=False, text_fmt='', limits=None,
    zero_center=True, text_vals=None, fps=1, figname=None, dpi=None, 
    bitrate=-1, text_kws={},
    label_kws={}, tick_kws={}, tickm_kws={}, grid_kws={}, history_kws={},
    **plot_kws
):
    """Plot the 4D phase space trajectory of a single particle."""
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
