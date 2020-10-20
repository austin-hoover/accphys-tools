"""
This module provides functions to visualize beam distributions.
"""

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft
from pandas.plotting._matplotlib.tools import _set_ticks_props
from . import envelope_analysis as ea

# Global variables
labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]


# Helper functions
def setup_corner_axes_3x3(limits, gap=0.1, figsize=(8,8), norm_labels=False):
    """Set up 'corner' of 4x4 grid of subplots:
    
    Does not include diagonals. It returns the axes 
    marked with 'X' in the following diagram:
    
    O O O O
    X O O O
    X X O O
    X X X O
    
    It is used to plot the 6 unique pairwise relationships between 4 variables. 
    For example, if our variables are a, b, c, and d, the subplots would 
    contain the following plots:

    b-a
    c-a c-b
    d-a d-b d-c
    
    Inputs
    ------
    limits : tuple
        (umax, upmax), where u can be x or y. umax is the maximum extent of
        the real-space plot windows, while upmax is the maximum extent of 
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
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    
    # Configure axis limits, ticks, and labels
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
    
    # Edit axes
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
    
    
def corner(
    coords,
    env_params=None,
    mm_mrad=False,
    samples=2500,
    limits=None,
    padding=0.5, 
    figsize=(8,8),
    gap=0.1,
    s=1,
    c=None,
    histtype='step',
    bins='auto',
    reduce_hist_height=0.75,
    plt_kws={},
    hist_kws={},
    figname=None,
    dpi=300,
):
    """Plot lower corner of scatter matrix of distribution, envelope, or both.
    
    Pandas scatter_matrix produces scatter plots of all the particles, 
    which can be a lot for 100k+ points. I wanted to plot a small sample 
    of the particles in the scatter plots while keeping all particles for 
    the histograms. I also wanted to have the same ranges for the dimensions 
    x,y and for the slopes x',y', so that the x-y and x'-y' plots would have 
    square dimensions.
        
    Inputs
    ------
    coords : NumPy array, shape (number_of_particles, 4)
        Transverse phase space coordinates x, x', y, y'.
    env_params : array-like
        Envelope parameters [a, b, a', b', e, f, e', f']. This is specific to 
        rotating self-consistent beams; it allows comparison of the beam 
        envelope to the distribution.
    mm_mrad : boolean
        If True, convert coordinates from m-rad to mm-mrad.
    samples : int
        Number of randomly sampled data points to use in scatter plots.
        Default: 2,500.
    limits : tuple
        Manually set the maximum rms position and slope of the distribution 
        (umax, upmax). If None, auto-ranging is performed.
    padding : float
        Fraction of umax and upmax to pad the axis ranges with. The edge of the 
        plot will be at umax * (1 + padding).
    figsize : tuple, 
        The x and y size of the figure.
    gap : float
        Width of the gap between the subplots.
    s : float
        Marker size.
    c : str
        Marker color.
    histtype : str
        Argument in matplotlib.pyplot.hist()
    bins : str
        Argument in matplotlib.pyplot.hist()
    reduce_hist_height : float
        Reduce the height of the histogram from its default value. Must be in
        range [0, 1]. The new height will be reduce_hist_height * old_height.
    plt_kws : dict
        Keyword arguments for matplotlib.pyplot.scatter.
    hist_kws : dict
        Keyword arguments for matplotlib.pyplot.hist.
    figname : str
        Name of saved figure. 
    dpi : int
        Dots per inch of saved figure.
        
        
    Returns
    -------
    axes : Matplotlib axes object
        4x4 array of Axes objects.
    """        
    # Get envelope data 
    if env_params is not None:
        env_params = np.array(env_params)
        env_data = ea.get_coords(env_params)
        
    # Convert from m-rad to mm-mrad
    coords = np.copy(coords)
    if mm_mrad:
        coords *= 1000
        if env_params is not None:
            env_data *= 1000 
            
    # Add user supplied plot settings to keyword arguments
    plt_kws['s'] = s
    plt_kws['c'] = c
    plt_kws['edgecolors'] ='none'
    hist_kws['histtype'] = histtype
    hist_kws['bins'] = bins
    hist_kws['color'] = c
    
    # Setup figure
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.subplots_adjust(wspace=gap, hspace=gap)
    
    # Take samples
    nparts = coords.shape[0]
    cdf = pd.DataFrame(coords, columns=['x','xp','y','yp'])
    cdf_sample = cdf.sample(samples) if nparts > samples else cdf
    coords_sample = cdf_sample.values
    
    # Plot data
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            if i > j: 
                ax.scatter(coords_sample[:,j], coords_sample[:,i], **plt_kws)
                if env_params is not None:
                    ax.plot(env_data[j], env_data[i], 'k')
            elif i == j:
                ax.hist(coords[:,j], **hist_kws)
    
    # Configure axis limits
    if limits is None: 
        umax = 2 * cdf.std()[['x','y']].max()
        upmax = 2 * cdf.std()[['xp','yp']].max()
    else:
        umax, upmax = limits
    umax_padded, upmax_padded = (1+padding)*umax, (1+padding)*upmax
    limits = [(-umax_padded, umax_padded),
            (-upmax_padded, upmax_padded)] * 2
    
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
                ax.set_xticklabels([''])
            if j != 0:
                ax.set_yticklabels([''])
            # set labels
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize='x-large')
            else:
                ax.set_ylabel('')
            if i == 3:
                ax.set_xlabel(labels[j], fontsize='x-large')
            else:
                ax.set_xlabel('')
                
    # Edit histogram height
    for i in range(4):
        ax = axes[i, i]
        new_ylim = (1.0 / reduce_hist_height) * ax.get_ylim()[1]
        ax.set_ylim(0, new_ylim)
            
    # Set axis label sizes and orientations
    _set_ticks_props(axes, xlabelsize=8, xrot=0, ylabelsize=8, yrot=0)
    fig.align_labels()
        
    # Save figure
    if figname is not None:
        plt.savefig(figname, dpi=dpi)
        
    return axes


def corner_envelope(
    params,
    init_params=None,
    mm_mrad=False,
    limits=None,
    fill=False,
    edgecolor='black',
    fillcolor='lightsteelblue',
    padding=0.5,
    gap=0.1,
    c='black',
    figsize=(6,6),
    norm_labels=False,
    leg_loc=(0,1),
    leg_labels=('Initial', 'Final'),
    tight_layout=False,
    figname=None,
    dpi=None,
):       
    """Plot the 6 transverse phase space ellipses of the beam.
    
    Inputs
    ------
    params : array-like
        Envelope parameters [a, b, a', b', e, f, e', f'].
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
    edgecolor : str
        The color of the ellipse boundary.
    fillcolor : str
        The color of the ellipse interior.
    figsize : tuple
        The x and y size of the figure.
    norm_labels : boolean
        If True, add '_n' to the axis labels. E.g. 'x' -> 'x_n'.
    leg_loc : tuple
        Location of subplot of the legend. The legend identifies linestyles 
        used for the initial and final beam ellipses. (0,1) corresponds to 
        first row, second column.
    leg_labels : tuple
        Labels if plotting two envelopes on top of one another. For example,
        the default is ('Initial', 'Final') if plotting the initial and final 
        envelopes. 
    figname : str
        Name of saved figure -> plt.savefig(figname, dpi=dpi).
    dpi : int
        Dots per inch of saved figure -> plt.savefig(figname, dpi=dpi).
        
    Returns
    -------
    axes : Matplotlib axes object
        3x3 array of Axes objects.
    """    
    # Get ellipse boundary data for x, x', y, y'
    def get_ellipse_data(params):
        data = ea.get_coords(params)
        if mm_mrad:
            data *= 1000
        return data
    data = get_ellipse_data(params)
    if init_params is not None:
        init_data = get_ellipse_data(init_params)
            
    xdata, ydata = data[:-1], data[1:]

    # Configure axis limits
    if limits is None:
        umax = max(max(data[0]), max(data[2]))
        upmax = max(max(data[1]), max(data[3]))
    else:
        umax, upmax = limits
    umax_padded, upmax_padded = (1+padding)*umax, (1+padding)*upmax
    limits = (umax_padded, upmax_padded)
                
    # Set up figure
    fig, axes = setup_corner_axes_3x3(limits, gap, figsize, norm_labels)

    # Plot data
    for i in range(3):
        for j in range(3):
            ax = axes[i,j]
            if i >= j: 
                ax.plot(xdata[j], ydata[i], '-', color=c, lw=1)
                if fill:
                    ax.fill(xdata[j], ydata[i], facecolor=fillcolor, lw=0)
                if init_params is not None:
                    xdata, ydata = init_data[:-1], init_data[1:]
                    ax.plot(xdata[j], ydata[i], '--', color=c, lw=1)
    # Add legend
    if init_params is not None:
        handles = []
        handles.append(plt.plot([], [], c=c, ls='--', marker='')[0])
        handles.append(plt.plot([], [], c=c, ls='-', marker='')[0])
        i, j = leg_loc
        axes[i,j].legend(handles, leg_labels, fontsize='large', loc='center')
    
    _set_ticks_props(axes, xlabelsize=8, xrot=0, ylabelsize=8, yrot=0)
    fig.align_labels()
    fig.set_tight_layout(tight_layout)
        
    # Save
    if figname is not None:
        plt.savefig(figname, dpi=dpi)

    return axes


def plot_fft(x, y, legend_kws={}, grid=True, figname=None):
    """Compute and plot the FFT of two signals x and y on the same figure.
    
    Uses SciPy fft package. Particularly useful for plotting the horizontal 
    tunes of a particle its turn-by-turn x and y coordinates.

    Let N be the number of samples, M=N/2, x(t)=signal as function of time, 
    and w(f) be the FFT of x. Then w[0] is zero frequency component, w[1:M] 
    are positive frequency components, and w[M:N] are negative frequency 
    components.
    
    Inputs
    ------
    x{y} : Numpy array, shape (n,): 
        Contains the x{y} coordinate at n time steps.
    legend_kws : dict
        Keyword arguments for the legend.
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








#### Scatter plot with color weighted by density

# from scipy.stats import gaussian_kde

# # Generate fake data
# df = dist.cdf_n.sample(10000)
# x = df['x'].values
# y = df['y'].values

# # Calculate the point density
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)

# # Sort the points by density, so that the densest points are plotted last
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]

# fig, ax = plt.subplots(figsize=(5,5))
# ax.scatter(x, y, c=z, s=4, edgecolor='')
# plt.show()
