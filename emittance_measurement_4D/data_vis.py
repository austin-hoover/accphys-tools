import sys
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def ancestor_folder_path(current_path, ancestor_folder_name):  
    parent_path = os.path.dirname(current_path)
    if parent_path == current_path:
        raise ValueError("Couldn't find ancestor folder.")
    if parent_path.split('/')[-1] == ancestor_folder_name:
        return parent_path
    return ancestor_folder_path(parent_path, ancestor_folder_name)

sys.path.append(ancestor_folder_path(os.path.abspath(__file__), 'accphys'))
from tools import utils


DEFAULT_COLORCYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']


def possible_points(transfer_mats, moments, slopes): 
    coords_list = []
    for M, (sig_xx, sig_yy, sig_xy) in zip(transfer_mats, moments):
        Minv = np.linalg.inv(M)
        coords = []
        for slope in slopes:
            vec = [np.sqrt(sig_xx), slope, np.sqrt(sig_yy), slope]
            vec = np.matmul(Minv, vec)
            coords.append(vec)
        coords_list.append(coords)
    return np.array(coords_list)


def rec_lines(ax, transfer_mats, moments, plane='x-xp', 
              norm_mat=None, slopes=None, **plt_kws):
    
    if slopes is None:
        slopes = [-100., 100.]
    
    V = norm_mat
    if V is None:
        V = np.identity(4)
    Vinv = np.linalg.inv(V)
    
    for coords in possible_points(transfer_mats, moments, slopes):
        coords = utils.apply(Vinv, coords)
        if plane == 'x-xp':
            ax.plot(coords[:, 0], coords[:, 1], **plt_kws)
        elif plane == 'y-yp':
            ax.plot(coords[:, 2], coords[:, 3], **plt_kws)
    return ax


def reconstruction_lines(ax, transfer_mats_dict, moments_dict, plane='x-xp', norm_mat=None, 
                         slopes=None, legend=False, legend_kws=None, **plt_kws):
        
    plt_kws.setdefault('lw', 0.75)
    
    if legend_kws is None:
        legend_kws = dict()
    legend_kws.setdefault('fontsize', 'small')
    legend_kws.setdefault('loc', (1.03, 0.0))
    legend_kws.setdefault('ncol', 1)
    
    ws_ids = sorted(list(transfer_mats_dict))
    colors = DEFAULT_COLORCYCLE[:len(ws_ids)]
    for ws_id, color in zip(ws_ids, colors):
        rec_lines(ax, transfer_mats_dict[ws_id], moments_dict[ws_id], 
                  plane, norm_mat, slopes, color=color, **plt_kws)
    if legend:
        lines = [Line2D([0], [0], color=color) for color in colors]
        ax.legend(lines, ws_ids, **legend_kws)