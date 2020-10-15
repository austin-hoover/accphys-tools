"""
This module contains functions to animate the evolution of a beam in phase 
space.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns
from pandas.plotting._matplotlib.tools import _set_ticks_props
from . import envelope_analysis as ea
from matplotlib.patches import Ellipse, transforms
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]


def setup_corner_axes_3x3(limits, gap=0.1, figsize=(8,8), norm_labels=False):
    """Set up 'corner' of 4x4 grid of subplots:
    
    Does not include diagonals. It returns the axes 
    marked with 'X' in the following diagram:
    
    O O O O
    X O O O
    X X O O
    X X X O
    
    Motivation is to plot the 6 unique pairwise relationships
    between 4 variables. For example, if our variables are a,
    b, c, and d, the subplots would contain the following 
    plots:
    
    ba
    ca cb
    da db dc
    
    Inputs
    ------
    limits : tuple, length 2
        (umax, upmax), where u can be x or y. umax is the maximum extent of the
        real-space plot windows, while upmax is the maximum extent of the 
        phase-space plot windows.
    gap : float
        The width of the gap between subplots.
    figsize : tuple, length 2
        The figure size: (size_x, size_y)
    norm_labels : bool
        Whether to add an 'n' subscript to axis labels.

    Returns
    -------
    axes : Matplotlib Axes object
        Array of the figure axes.
    """
    # Create figure
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    plt.close()
    
    # Limits, ticks, and labels
    labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
    if norm_labels:
        labels = [r"$x_n$", r"$x_n'$", r"$y_n$", r"$y_n'$"]
    umax, upmax = limits
    limits = [(-umax, umax), (-upmax, upmax)] * 2
    utick, uptick = umax * 0.8, upmax * 0.8
    ticks = [[-utick, 0, utick], [-uptick, 0, uptick]] * 2
    ticks = np.around(ticks, decimals=1)
    xlimits = limits[:-1]
    xticks = ticks[:-1]
    xlabels = labels[:-1]
    ylimits = limits[1:]
    yticks = ticks[1:]
    ylabels = labels[1:]
    
    for i in range(3):
        for j in range(3):
            ax = axes[i,j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(xticks[j])
            ax.set_yticks(yticks[i])
            ax.set_xlim(xlimits[j])
            ax.set_ylim(ylimits[i])
            if i < j:
                ax.axis('off')
            if j == 0:
                ax.set_ylabel(ylabels[i], fontsize='x-large')
            if i == 2:
                ax.set_xlabel(xlabels[j], fontsize='x-large')
    
    return fig, axes


def setup_corner_axes_no_diag(limits, gap=0.1, figsize=(8,8)):
    """Set up axes for corner plot (no diagonal).
    
    limits: tuple - (umax, upmax)
    """
    # Create figure
    fig, axes = plt.subplots(4, 4, sharex='col', sharey='row', figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    plt.close()
    # visibility of axes
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i <= j:
                ax.axis('off')
    # Limits, ticks, and labels
    labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
    if norm_labels:
        labels = [r"$x_n$", r"$x_n'$", r"$y_n$", r"$y_n'$"]
    umax, upmax = limits
    limits = [(-umax, umax), (-upmax, upmax)] * 2
    utick, uptick = umax * 0.8, upmax * 0.8
    ticks = np.around([[-utick, 0, utick], [-uptick, 0, uptick]] * 2, decimals=1)
    for k in range(4):
        axes[3,k].set_xlim(limits[k])
        axes[3,k].set_xticks(ticks[k])
        axes[3,k].set_xlabel(labels[k], fontsize='x-large')
        axes[k,0].set_ylim(limits[k])
        axes[k,0].set_yticks(ticks[k])
        axes[k,0].set_ylabel(labels[k], fontsize='x-large')
    return fig, axes


def corner(
    coords_list,
    nturns,
    samples=5000,
    limits=None,
    padding=0.5,
    s=1.0,
    alpha=1.0,
    figsize=(7,7),
    gap=0.1,
    hist_spines=False,
    reduce_hist_height=0.75,
    norm_labels=False,
    figname=None,
    dpi=None,
    fps=1,
    bitrate=None
):
    """Animate corner plots of TBT coordinate data.
    
    cdfs: list of coordinate DataFrames
    """
    
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
    
    # Auto limits
    if limits is None:
        initial_beam = cdfs[0]
        umax = 2 * initial_beam.std()[['x','y']].max()
        upmax = 2 * initial_beam.std()[['xp','yp']].max()
    else:
        umax, upmax = limits
    umax_padded, upmax_padded = (1+padding)*umax, (1+padding)*upmax
    limits = [(-umax_padded, umax_padded),
              (-upmax_padded, upmax_padded)] * 2
   
    # Ticks and labels
    labels = ['x', 'xp', 'y', 'yp']
    scale = 1.0
    utick, uptick = umax * scale, upmax * scale
    ticks = np.around([[-utick, 0, utick], [-uptick, 0, uptick]] * 2, 
                      decimals=1)
    for k in range(4):
        axes[k,0].set_yticks(ticks[k])
        axes[3,k].set_xticks(ticks[k])
        axes[k,0].set_ylabel(labels[k], fontsize='large')
        axes[3,k].set_xlabel(labels[k], fontsize='large')

    # Set limits
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            if i != j:
                ax.set_xlim(limits[j])
                ax.set_ylim(limits[i])
            else:
                ax.set_xlim(limits[j])
            # turn off ticks for non-edge subplots
            if (j != 0): ax.yaxis.set_visible(False)
            if (i != 3): ax.xaxis.set_visible(False)
            
    # Turn on/off borders for histogram subplot windows
    if not hist_spines:
        for i in range(4):
            for j in range(4):
                ax = axes[i,j]
                if i < j:
                    ax.axis('off')
                elif i == j:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    if i == 0:
                        ax.spines["left"].set_visible(False)
                        ax.set_ylabel('')
                        ax.set_yticks([])
                else:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                        

    # Axis label sizes and orientations
    _set_ticks_props(axes, xlabelsize=8, xrot=0, ylabelsize=8, yrot=0)

    # Create array of line2D objects
    lines = [[],[],[],[]]
    for i in range(4):
        for j in range(4):
            if i > j:
                line, = axes[i,j].plot([], [], 'o',
                                    alpha=alpha,
                                    markersize=s,
                                    markeredgewidth=0,
                                    fillstyle='full')
                lines[i].append(line)

    # Initialization function: plot the background of each frame.
    def init():
        for i in range(4):
            for j in range(4):
                if i > j:
                    lines[i][j].set_data([], [])
        return [line for row in lines for line in row],

    # Animation function to be called sequentially.
    def animate(i):
        df = pd.DataFrame(coords_list[i])
        df_samp = df.sample(samples) if samples < df.shape[0] else df
        for r in range(4):
            for c in range(4):
                ax = axes[r, c]
                if r > c:
                    lines[r][c].set_data(df_samp.iloc[:,c].values,
                                         df_samp.iloc[:,r].values)
                elif r == c:
                    ax.cla() # This clears everything. 
                             # Can I just clear the data?
                    ax.set_xlim(limits[r])
                    ax.set_xticks(ticks[r])
                    if r == 0:
                        ax.set_yticks([])
                    ax.set_xlabel(labels[r])
                    ax.set_ylabel('')
                    ax.hist(df.iloc[:,r], bins='auto', histtype='step')
                else:
                    ax.axis('off')
                    
        # Reset tick sizes (they have been cleared)
        _set_ticks_props(axes, xlabelsize=8, xrot=0, ylabelsize=8, yrot=0)
        
        # Edit histogram height
        for j in range(4):
            ax = axes[j, j]
            new_ylim = (1.0 / reduce_hist_height) * ax.get_ylim()[1]
            ax.set_ylim(0, new_ylim)
        
        # Show turn number
        [t.set_visible(False) for t in extra_ax.texts]
        extra_ax.text(0.5, 0.5, 'turn {}'.format(i))

        return [line for row in lines for line in row]

    # Call the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nturns, interval=1000/fps)
    # Save animation as mp4
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi, progress_callback=None)
    return anim


def corner_onepart(
    coords,
    limits=None,
    psi=0.0,
    padding=0.75,
    c='black',
    s=5,
    update_string='Turn = {}',
    show_history=False,
    positions=None,
    figsize=(7, 7),
    gap=0.1,
    norm_labels=False,
    fps=5,
    figname=None,
    dpi=None,
):
    """Animation of single particle coordinates."""
    
    # Split the data
    hdata, vdata = coords[:, :-1], coords[:, 1:]
    
    # Configure axis limits, ticks, and labels
    labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
    if norm_labels:
        labels = [l + '_n' for l in labels]
        
    if limits is None:
        x, xp, y, yp = coords.T
        umax = max(max(x), max(y))
        upmax = max(max(xp), max(yp))
    else:
        umax, upmax = limits
    umax_padded, upmax_padded = (1+padding)*umax, (1+padding)*upmax
    limits = 2 * [(-umax_padded, umax_padded), (-upmax_padded, upmax_padded)]
    
    utick, uptick = umax_padded * 0.8, upmax_padded * 0.8
    ticks = np.around(
        2 * [[-utick, 0, utick], [-uptick, 0, uptick]], decimals=1
    )
    xlimits, ylimits = limits[:-1], limits[1:]
    xticks, yticks = ticks[:-1], ticks[1:]
    xlabels, ylabels = labels[:-1], labels[1:]

    # Create figure
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    plt.close()
    
    lines = [[],[],[]]
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            line, = ax.plot([], [], 'o', lw=0, ms=s, color=c)
            lines[i].append(line)
            
    for i in range(3):
        for j in range(3):
            ax = axes[i,j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(xticks[j])
            ax.set_yticks(yticks[i])
            ax.set_xlim(xlimits[j])
            ax.set_ylim(ylimits[i])
            if i < j:
                ax.axis('off')
            if j == 0:
                ax.set_ylabel(ylabels[i], fontsize='x-large')
            if i == 2:
                ax.set_xlabel(xlabels[j], fontsize='x-large')

    def update(t):
        if positions is not None:
            t = positions[t]
        axes[1,1].set_title(update_string.format(t))
        for i in range(3):
            for j in range(3):
                if i >= j:
                    if show_history:
                        lines[i][j].set_data(hdata[:t+1, j], vdata[:t+1, i])
                    else:
                        lines[i][j].set_data(hdata[t, j], vdata[t, i])
                        
                        
    # Call animator
    anim = animation.FuncAnimation(fig, update, frames=coords.shape[0],
                                   interval=1000/fps)
    # Save animation
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi, progress_callback=None)

    return anim


def corner_envelope(
    params,
    limits=None,
    padding=0.75,
    c='lightsteelblue',
    fill=True,
    show_initial=False,
    clear_history=True,
    plot_boundary=True,
    update_string='Turn = {}',
    positions=None,
    figsize=(8, 8),
    gap=0.1,
    norm_labels=False,
    fps=5,
    figname=None,
    dpi=None,
):
    """Corner plot animation of beam envelope.

    The bounding 4D ellipse of a rotating self-consistent beam can be
    parameterized as:

        x = a cos(psi) + b sin(psi)
        x' = a' cos(psi) + b' sin(psi)
        y = e cos(psi) + e sin(psi)
        y' = e' cos(psi) + e' sin(psi)

    See reference [1] for more details. The parameters a, b, e, f, and
    their derivatives define ellipses for the six pairwise relationships
    between x, x', y, and y'.

    Inputs
    ------
    params : NumPy array, shape (nturns, 8)
        Columns are [a,b,a',b',e,f,e',f']. Rows are frame number.
    limits : tuple
        Manually set the maximum rms position and slope of the distribution
        (umax, upmax). If None, auto-ranging is performed.
    padding : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of the
        plot will be at umax * (1 + padding).
    c : str
        Color of ellipse boundaries.
    fill : bool
        Wheter to fill the ellipse.
    show_initial : bool
        Whether to show static initial envelope in background.
    clear_history : bool
        Whether to clear the previous frames before plotting the new frame.
    plot_boundary : bool
        Whether to plot the ellipse boudary.
    update_string : str
        Each new frame will display text indicating the turn or position. For
        example: 'Turn = 5' or 's = 17.4 m'. The string that will be printed
        is update_string.format(t), where t is a turn number or position.
    positions : array like
        Array which maps frame number to longitudinal position s.
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

    References
    ----------
    [1] Danilov, V., Cousineau, S., Henderson, S., and Holmes, J., Phys. Rev.
        ST Accel. Beams 6, 094202 (2003).
    """
    # Get ellipse boundary data for x, x', y, y'
    nturns = params.shape[0]
    n_angles = 50
    tracked = np.zeros((nturns, n_angles, 4))
    for i in range(nturns):
        x, xp, y, yp = ea.get_coords(params[i], n_angles)
        tracked[i] = np.vstack([x, xp, y, yp]).T

    # Configure axis limits, ticks and labels
    labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
    if norm_labels:
        labels = [l + '_n' for l in labels]
        
    if limits is None:
        x, xp, y, yp = tracked[0].T
        umax = max(max(x), max(y))
        upmax = max(max(xp), max(yp))
    else:
        umax, upmax = limits
    umax_padded, upmax_padded = (1+padding)*umax, (1+padding)*upmax
    limits = 2 * [(-umax_padded, umax_padded), (-upmax_padded, upmax_padded)]
    
    utick, uptick = umax_padded * 0.8, upmax_padded * 0.8
    ticks = np.around(
        2 * [[-utick, 0, utick], [-uptick, 0, uptick]], decimals=1
    )
    xlimits, ylimits = limits[:-1], limits[1:]
    xticks, yticks = ticks[:-1], ticks[1:]
    xlabels, ylabels = labels[:-1], labels[1:]

    # Create figure
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    plt.close()
    
    lines = [[],[],[]]
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            line, = ax.plot([], [], 'k-', lw=1)
            lines[i].append(line)
            
    for i in range(3):
        for j in range(3):
            ax = axes[i,j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(xticks[j])
            ax.set_yticks(yticks[i])
            ax.set_xlim(xlimits[j])
            ax.set_ylim(ylimits[i])
            if i < j:
                ax.axis('off')
            if j == 0:
                ax.set_ylabel(ylabels[i], fontsize='x-large')
            if i == 2:
                ax.set_xlabel(xlabels[j], fontsize='x-large')
        
    # Get initial data
    data_init = tracked[0]
    xdata_init, ydata_init = data_init[:, :-1], data_init[:, 1:]

    def update(t):

        if clear_history:
            [patch.remove() for ax in axes.flatten() for patch in ax.patches]
        
        data = tracked[t]
        xdata, ydata = data[:, :-1], data[:, 1:]
        if positions is not None:
            t = positions[t]
        axes[1,1].set_title(update_string.format(t))
        for i in range(3):
            for j in range(3):
                if i >= j:
                    ax = axes[i, j]
                    if fill:
                        ax.fill(xdata[:,j], ydata[:,i], facecolor=c, lw=0)
                    if plot_boundary:
                        lines[i][j].set_data(xdata[:, j], ydata[:, i])
                    if show_initial and clear_history:
                        ax.plot(xdata_init[:,j], ydata_init[:,i], 'k--', lw=0.5, alpha=0.25)
                        
    # Call animator
    anim = animation.FuncAnimation(fig, update, frames=nturns, 
                                   interval=1000/fps)
    # Save animation
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi, progress_callback=None)

    return anim
    

def xy_envelope(
    params,
    umax=None,
    padding=0.75,
    c='tab:blue',
    fill=True,
    clear_history=True,
    alpha=1.0,
    show_initial=False,
    update_string='Turn = {}',
    positions=None,
    figsize=(4, 4),
    gap=0.1,
    norm_labels=False,
    fps=5,
    figname=None,
    dpi=None,
):
    """Plot the ellipse in real space."""
        
    # Get ellipse boundary data for x and y
    nturns = params.shape[0]
    psi = np.linspace(0.0, 2*np.pi, 50)
    cos, sin = np.cos(psi), np.sin(psi)
    x, y = [], []
    for i in range(nturns):
        a, b, ap, bp, e, f, ep, fp = params[i]
        x.append(a*cos + b*sin)
        y.append(e*cos + f*sin)
            
    # Configure axis limits
    if not umax:
        umax = max(max(x[0]), max(y[0]))
    umax_padded = (1 + padding) * umax

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.close()
        
    def reset_axes():
        scale = 1.0
        ax.set_xticks([-scale*umax, 0, scale*umax])
        ax.set_yticks([-scale*umax, 0, scale*umax])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-umax_padded, umax_padded)
        ax.set_ylim(-umax_padded, umax_padded)
        ax.set_xlabel('x', fontsize='x-large')
        ax.set_ylabel('y', fontsize='x-large')
        if show_initial:
            ax.plot(x[0], y[0], 'k--', lw=1)
    
    def update(t):
        if clear_history:
            ax.clear()
        phi = -ea.tilt_angle(params[t], degrees=True)
        cx, cy = ea.radii(params[t])
        ellipse = Ellipse((0, 0), 2*cx, 2*cy, fill=fill, facecolor=c, 
                          alpha=alpha, edgecolor='k', lw=2)
        transf = transforms.Affine2D().rotate_deg(phi)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

        if positions is not None:
            t = positions[t]
            ax.set_title(update_string.format(t))

        reset_axes()

    # Call animator
    anim = animation.FuncAnimation(fig, update, frames=nturns, interval=1000/fps)
    
    # Save animation
    writer = animation.writers['ffmpeg'](fps=fps)
    if figname is not None:
        anim.save(figname, writer=writer, dpi=dpi, progress_callback=None)

    return anim
