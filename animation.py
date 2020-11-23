"""
This module contains functions to animate the evolution of a beam of
particles in phase space.

TO DO
------
* Plot multiple envelopes at once.
* Write function specifically for single particle motion which shows the
  coordinate vector as an arrow.
"""

# 3rd party
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns
from pandas.plotting._matplotlib.tools import _set_ticks_props
from matplotlib.patches import Ellipse, transforms

# My modules
from . import envelope_analysis as ea
from .plotting import setup_corner_axes_3x3, get_u_up_max, get_u_up_max_global
from .utils import add_to_dict

# Settings
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

# Module level variables
labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
    

def corner(
    coords,
    samples=2000,
    skip=0,
    limits=None,
    padding=0.25,
    figsize=(7, 7),
    gap=0.1,
    s=2.0,
    c='tab:blue',
    hist=True,
    hist_spines=False,
    hist_height=0.7,
    plt_kws={},
    hist_kws={},
    update_str='Turn = {}',
    update_vals=None,
    norm_labels=False,
    figname=None,
    dpi=None,
    fps=1,
    env=None,
    c_env='k',
):
    """Frame-by-frame phase space projections of the beam.
    
    Parameters
    ----------
    coords : list or NumPy array, shape (nturns, nparts, 4)
        Each element of `coords` contains the coordinate array at a particular
        frame.
    samples : int
        The number of particles to use in the scatter plots.
    skip : int
        The coordinates will be plotted every `skip` + 1 frames.
    limits : tuple
        (`umax`, `upmax`), where u can be x or y. `umax` is the maximum extent
        of the real-space plot windows, while `upmax` is the maximum extent of
        the phase-space plot windows. If None, they are set automatically.
    padding : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of the
        plot will be at umax * (1 + padding).
    figsize : tuple,
        The x and y size of the figure.
    gap : float
        Width of the gap between the subplots.
    s : float
        The marker size.
    c : str
        The marker color.
    hist : bool
        Whether to plot histograms on the diagonals.
    hist_height : float
        Reduce the height of the histogram from its default value. Must be in
        range [0, 1]. The new height will be hist_height * old_height.
    plt_kws : dict
        Keyword arguments for matplotlib.pyplot.scatter. It will override the
        `s` and `c` parameters.
    hist_kws : dict
        Keyword arguments for matplotlib.pyplot.hist. It will override the
        `histtype` and `bins` arg
    update_str : str
        Each new frame will display text indicating the turn or position. For
        example: 'Turn = 5' or 's = 17.4 m'. The string that will be printed
        is update_str.format(f), where f = update_vals[t] and t is the frame
        number.
    update_vals : array like
        Array which maps frame number to a value such as the s position.
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
    figname : str
        If not None, calls animation.save(figname).
    fps : int
        Frames per second of the animation.
    dpi : int
        DPI for saved animation.
    env : NumPy array, shape (nframes, 8)
        The beam envelope parameters, which are used to plot the beam envelope
        if provided.
    c_env : str
        The color of the beam envelope, if provided.
    """
    # Add user supplied keyword arguments
    add_to_dict(plt_kws, 'ms', s)
    add_to_dict(plt_kws, 'color', c)
    add_to_dict(plt_kws, 'marker', '.')
    add_to_dict(plt_kws, 'markeredgewidth', 0)
    add_to_dict(plt_kws, 'lw', 0)
    add_to_dict(plt_kws, 'fillstyle', 'full')
    add_to_dict(hist_kws, 'histtype', 'step')
    add_to_dict(hist_kws, 'bins', 'auto')
    add_to_dict(hist_kws, 'color', c)
        
    # Setup
    if type(coords) is list:
        coords = np.array(coords)
    nframes = coords.shape[0]
    if len(coords.shape) == 2: # single particle bunch
        coords = coords[:, np.newaxis, :]
    if update_vals is None:
        update_vals = list(range(nframes))
    # Skip frames
    plot_every = skip + 1
    coords = coords[::plot_every]
    update_vals = update_vals[::plot_every]
    nframes = coords.shape[0]
    
    # Setup figure
    plt.clf()
    fig, axes = plt.subplots(4, 4, sharex='col', figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    plt.close()
    
    # Add subplot for text
    extra_ax = fig.add_subplot(447)
    [extra_ax.spines[i].set_visible(False) for i in extra_ax.spines]
    extra_ax.set_xticks([])
    extra_ax.set_yticks([])
    
    # Take random sample of the coordinates
    coords_samp, (nframes, nparts, ndims) = coords, coords.shape
    if nparts > samples:
        idx = np.random.choice(nparts, samples, replace=False)
        coords_samp = coords[:, idx, :]
    
    # Configure axis limits
    if limits is None:
        umax, upmax = get_u_up_max_global(coords)
    else:
        umax, upmax = limits
    umax_pad, upmax_pad = (1+padding)*umax, (1+padding)*upmax
    limits = 2 * [(-umax_pad, umax_pad), (-upmax_pad, upmax_pad)]
    
    # Get envelope coordinates
    env_coords = None
    if env is not None:
        env_coords = ea.get_ellipse_coords(env)
        env_coords = env_coords[::plot_every]
    
    if not hist:
        return corner_nohist(
            coords_samp, (umax_pad, upmax_pad), padding, figsize, gap,
            plt_kws, update_str, update_vals, norm_labels, figname, dpi,
            fps, env_coords, c_env)

    # Set axis limits and spines
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            [ax.spines[s].set_visible(False) for s in ['top', 'right']]
            if i > j:
                ax.set_xlim(limits[j])
                ax.set_ylim(limits[i])
            elif i == j:
                ax.set_xlim(limits[j])
                ax.yaxis.set_visible(False) # no yticks for histograms
                if not hist_spines or i == 0:
                    ax.spines["left"].set_visible(False)
                    ax.set_ylabel('')
                    ax.set_yticks([])
            else:
                ax.axis('off')
                    
    # Ticks and labels
    labels = ['x', 'xp', 'y', 'yp']
    scale = 1.0
    utick, uptick = umax * scale, upmax * scale
    ticks = 2 * [[-utick, 0, utick], [-uptick, 0, uptick]]
    ticks = np.round(ticks, decimals=1)
    for i in range(4):
        for j in range(i):
            ax = axes[i, j]
            if j == 0:
                ax.set_yticks(ticks[i])
                ax.set_ylabel(labels[i], fontsize='large')
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')
            if i == 3:
                ax.set_xticks(ticks[j])
                ax.set_xlabel(labels[j], fontsize='large')
            else:
                ax.set_xticklabels([])
                ax.set_xlabel('')
                        
    # Axis ticklabels - sizes and orientations
    _set_ticks_props(axes, xlabelsize=8, xrot=0, ylabelsize=8, yrot=0)

    # Create array of line2D objects
    lines = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i > j:
                line, = ax.plot([], [], **plt_kws)
                lines[i].append(line)
                
    # Beam envelope
    if env is not None:
        lines_env = [[], [], [], []]
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                if i > j:
                    line, = ax.plot([], [], '-', color=c_env)
                    lines_env[i].append(line)

    def init():
        """Plot the background of each frame."""
        for i in range(4):
            for j in range(4):
                if i > j:
                    lines[i][j].set_data([], [])
        return [line for row in lines for line in row]

    def update(t):
        """Animation function to be called sequentially."""
        X, X_samp = coords[t], coords_samp[t]
        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                if i > j:
                    lines[i][j].set_data(X_samp[:, j], X_samp[:, i])
                    if env is not None:
                        X_env = env_coords[t]
                        lines_env[i][j].set_data(X_env[:, j], X_env[:, i])
                elif i == j:
                    # Clear the axis. This clears everything... can I just
                    # clear the data?
                    ax.cla()
                    ax.set_xlim(limits[i])
                    ax.set_xticks(ticks[i])
                    if i == 0:
                        ax.set_yticks([])
                    if j == 3:
                        ax.set_xlabel(labels[i])
                    ax.set_ylabel('')
                    ax.hist(X[:, i], **hist_kws)
                    
        # Reset tick sizes (they have been cleared)
        _set_ticks_props(axes, xlabelsize=8, xrot=0, ylabelsize=8, yrot=0)
        
        # Edit histogram height
        for k in range(4):
            ax = axes[k, k]
            new_ylim = ax.get_ylim()[1] / hist_height
            ax.set_ylim(0, new_ylim)
        
        # Display update
        if update_vals is not None:
             t = update_vals[t]
        [text.set_visible(False) for text in extra_ax.texts]
        extra_ax.text(0.5, 0.5, update_str.format(t))
        return [line for row in lines for line in row]

    # Call animator and (maybe) save the animation
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=nframes, interval=1000/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi)
    return anim
    
    
def corner_nohist(
    coords,
    limits=(1, 1),
    padding=0.25,
    figsize=(7, 7),
    gap=0.1,
    plt_kws={},
    update_str='Turn = {}',
    update_vals=None,
    norm_labels=False,
    figname=None,
    dpi=None,
    fps=1,
    env_coords=None,
    c_env='k',
):
    """Corner plot without histograms on the diagonals.
    
    Do not call directly... use `corner` with `hist=False`.
    """
    # Create figure
    fig, axes = setup_corner_axes_3x3(limits, gap, figsize, norm_labels)
    plt.close()
    
    # Create list of Line2D objects
    lines = [[], [], []]
    for i in range(3):
        for j in range(3):
            line, = axes[i, j].plot([], [], **plt_kws)
            lines[i].append(line)
            
    # Beam envelope
    if env_coords is not None:
        lines_env = [[], [], []]
        for i in range(3):
            for j in range(3):
                line, = axes[i, j].plot([], [], '-', color=c_env)
                lines_env[i].append(line)
         
    def update(t):
        """Animation function to be called sequentially."""
        X = coords[t]
        hdata, vdata = X[:, :-1], X[:, 1:]
        if env_coords is not None:
            X_env = env_coords[t]
            hdata_env, vdata_env = X_env[:, :-1], X_env[:, 1:]
        for i in range(3):
            for j in range(3):
                if i >= j:
                    lines[i][j].set_data(hdata[:, j], vdata[:, i])
                    if env_coords is not None:
                        lines_env[i][j].set_data(hdata_env[:, j],
                                                 vdata_env[:, i])
        # Display update
        if update_vals is not None:
            t = update_vals[t]
        axes[1, 1].set_title(update_str.format(t))
        return [line for row in lines for line in row]
        
    # Call animator and (maybe) save the animation
    anim = animation.FuncAnimation(fig, update, init_func=None,
                                   frames=coords.shape[0], interval=1000/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi)
    return anim
    

def corner_env(
    params,
    limits=None,
    skip=0,
    padding=0.05,
    edgecolor='k',
    facecolor='lightsteelblue',
    fill=True,
    show_init=False,
    clear_history=True,
    plot_boundary=True,
    update_str='Turn = {}',
    update_vals=None,
    figsize=(8, 8),
    gap=0.1,
    norm_labels=False,
    fps=5,
    figname=None,
    dpi=None,
):
    """Corner plot animation of beam envelope.

    The bounding 4D ellipse of the beam can be parameterized as
        x = a cos(psi) + b sin(psi), x' = a' cos(psi) + b' sin(psi),
        y = e cos(psi) + e sin(psi), y' = e' cos(psi) + e' sin(psi),
    where 0 <= psi <= 2pi.

    Inputs
    ------
    params : list or NumPy array, shape (nframes, 8)
        Columns are [a, b, a', b', e, f, e', f'].
    limits : tuple
        (`umax`, `upmax`), where u can be x or y. `umax` is the maximum extent
        of the real-space plot windows, while `upmax` is the maximum extent of
        the phase-space plot windows. If None, they are set automatically.
    skip : int
        The coordinates will be plotted every `skip` + 1 frames.
    padding : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of
        the plot will be at umax * (1 + padding).
    edgecolor : str
        Color of ellipse boundary.
    facecolor : str
        Color of the ellipse interior.
    fill : bool
        Whether to fill the ellipse.
    show_init : bool
        Whether to show static initial envelope in background.
    clear_history : bool
        Whether to clear the previous frames before plotting the new frame.
    plot_boundary : bool
        Whether to plot the ellipse boudary.
    update_str : str
        Each new frame will display text indicating the turn or position. For
          example: 'Turn = 5' or 's = 17.4 m'. The string that will be printed
          is update_str.format(f), where f = update_vals[t] and t is the frame
          number.
    update_vals : array like
        Array which maps frame number to a value such as the s position.
    figsize : tuple
        Size of figure, (x_size, y_size).
    gap : float
        Size of the gap between subplots.
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels. Ex: 'x' --> 'x_n'.
    fps : int
        Frames per second of the animation.
    figname : str
        If not None, calls animation.save(figname).
    dpi : int
        DPI for saved animation.

    Returns
    -------
    anim : matplotlib.animation object
    """
    # Get ellipse coordinates
    coords = ea.get_ellipse_coords(params)
    nframes = coords.shape[0]
    if update_vals is None:
        update_vals = list(range(nframes))
        
    # Skip frames
    coords = coords[::skip+1]
    update_vals = update_vals[::skip+1]
    nframes = coords.shape[0]
        
    # Store initial ellipse
    X_init = coords[0]
    hdata_init, vdata_init = X_init[:, :-1], X_init[:, 1:]
        
    # Configure axis limits
    if limits is None:
        umax, upmax = get_u_up_max_global(coords)
    else:
        umax, upmax = limits
    limits = ((1 + padding)*umax, (1 + padding)*upmax)

    # Create figure
    fig, axes = setup_corner_axes_3x3(limits, gap, figsize, norm_labels)
    plt.close()
    
    # Create list of Line2D objects
    lines = [[], [], []]
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            line, = ax.plot([], [], '-', color=edgecolor, lw=2)
            lines[i].append(line)

    def update(t):
        """Animation function to be called sequentially."""
        if clear_history:
            [patch.remove() for ax in axes.ravel() for patch in ax.patches]
        X = coords[t]
        hdata, vdata = X[:, :-1], X[:, 1:]
        for i in range(3):
            for j in range(3):
                if i >= j:
                    ax = axes[i, j]
                    if fill:
                        ax.fill(hdata[:,j], vdata[:,i],
                                facecolor=facecolor, lw=0)
                    if plot_boundary:
                        lines[i][j].set_data(hdata[:, j], vdata[:, i])
                    if show_init and clear_history:
                        ax.plot(hdata_init[:,j], vdata_init[:,i], 'k--',
                                lw=0.5, alpha=0.25)
        # Display update
        if update_vals is not None:
            t = update_vals[t]
        axes[1, 1].set_title(update_str.format(t))
                        
    # Call animator and (maybe) save the animation
    anim = animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=1000/fps)
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi, progress_callback=None)

    return anim
