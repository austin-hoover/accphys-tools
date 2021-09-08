import numpy as np
from matplotlib import pyplot as plt


def scatter_hist(x, y, ax, ax_marg_x, ax_marg_y, joint_kws=None, marginal_kws=None):
    if joint_kws is None:
        joint_kws = dict()
    if marginal_kws is None:
        marginal_kws = dict()        
    if 'range' in joint_kws:
        xrange, yrange = joint_kws['range']
    else:
        xrange = yrange = None
    if 'bins' in joint_kws:
        bins = joint_kws['bins']
    else:
        heights, edges, patches = ax_marg_x.hist(x, bins='auto', range=xrange, **marginal_kws)
        for patch in patches:
            patch.set_visible(False)
        bins = len(heights)
        joint_kws['bins'] = bins
    ax_marg_x.hist(x, range=xrange, bins=bins, **marginal_kws)
    ax_marg_y.hist(y, range=yrange, bins=bins, orientation='horizontal', **marginal_kws)
    ax.hist2d(x, y, **joint_kws)
    return ax


def create_grid(fig, gridspec, row, col):
    ax_joint = fig.add_subplot(gridspec[row, col])
    ax_marg_x = fig.add_subplot(gridspec[row - 1, col])
    ax_marg_y = fig.add_subplot(gridspec[row, col + 1])
    for ax in [ax_marg_x, ax_marg_y]:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(False)
    return ax_joint, ax_marg_x, ax_marg_y


def emittances_joint_hist(emittances, lims=((10, 40), (0, 20)), joint_kws=None, marginal_kws=None):
    
    if joint_kws is None:
        joint_kws = dict()
    if marginal_kws is None:
        marginal_kws = dict()
        
    joint_kws.setdefault('cmap', 'fire_r')
    joint_kws.setdefault('bins', 75)
    joint_kws['range'] = lims
    marginal_kws.setdefault('histtype', 'step')
    marginal_kws.setdefault('color', 'black')
    
    fig = plt.figure(figsize=(10, 4))
    h = 1.5
    gridspec = fig.add_gridspec(2, 5, width_ratios=(7, h, 2.5, 7, h), height_ratios=(h, 7),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0, hspace=0)
    ax1, ax1_marg_x, ax1_marg_y = create_grid(fig, gridspec, 1, 0)
    ax2, ax2_marg_x, ax2_marg_y = create_grid(fig, gridspec, 1, 3)
    
    scatter_hist(emittances[:, 0], emittances[:, 1], ax1, ax1_marg_x, ax1_marg_y, 
                 joint_kws, marginal_kws)
    scatter_hist(emittances[:, 2], emittances[:, 3], ax2, ax2_marg_x, ax2_marg_y, 
                 joint_kws, marginal_kws)
    ax1_marg_x.set_xlim(lims[0])
    ax1_marg_y.set_ylim(lims[1])
    ax2_marg_x.set_xlim(lims[0])
    ax2_marg_y.set_ylim(lims[1])
    ax1.set_xlabel(r'$\varepsilon_x$ [mm mrad]')
    ax1.set_ylabel(r'$\varepsilon_y$ [mm mrad]')
    ax2.set_xlabel(r'$\varepsilon_1$ [mm mrad]')
    ax2.set_ylabel(r'$\varepsilon_2$ [mm mrad]')
    ax1_marg_x.set_title(r'Apparent emittances ($\varepsilon_x$, $\varepsilon_y$)')
    ax2_marg_x.set_title(r'Intrinsic emittances ($\varepsilon_1$, $\varepsilon_2$)')      
    return ax1, ax1_marg_x, ax1_marg_y, ax2, ax1_marg_y, ax1_marg_y