"""
Error vs. beam intensity.
"""
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pprint import pprint

from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode, WireScannerNode, add_analysis_node
from orbit.analysis.analysis import intrinsic_emittances, apparent_emittances
from orbit.envelope import Envelope
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

sys.path.append('/Users/46h/Research/code/accphys/pyorbit/measurement')
from utils import PhaseController
from data_analysis import reconstruct


# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # GeV/c^2
kin_energy = 0.8 # GeV
bunch_length = 160.0 # m
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
latfile = '_input/rtbt.lat'
latseq = 'surv'

# Initial beam
beam_type = 'danilov'
eps = 40e-6 # nonzero intrinsice emittance ex + ey [m*rad]
mode = 1
ex_frac = 0.4
ex, ey = ex_frac * eps, (1 - ex_frac) * eps
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)

# Scan parameters
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
steps_per_dim = 6 # number of steps for each dimension
method = 2
phase_coverage = 180 # [deg]
max_betas = (40, 40) # (x, y)


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

# Get correct magnet settings
dummy_lattice = hf.lattice_from_file(latfile, latseq)
controller = PhaseController(dummy_lattice, init_twiss, mass, kin_energy, ref_ws_name)
scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
quad_strengths_list = np.load('quad_strengths_list.npy')
# quad_strengths_list = []
# for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
#     fstr = 'Setting phases (scan {}/{}): nux, nuy = {:.3f}, {:.3f}.'
#     print fstr.format(scan_index, 2 * steps_per_dim, nux, nuy)
#     quad_strengths = controller.set_ref_ws_phases(nux, nuy, verbose=2)
#     quad_strengths_list.append(quad_strengths)

# Intial envelope
init_env = Envelope(eps, mode, ex_frac, mass, kin_energy, length=bunch_length)
ax0, ay0, bx0, by0 = init_twiss
init_env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)
init_env.set_twiss_param_4D('nu', np.radians(60))

# Create lattice
lattice = hf.lattice_from_file(latfile, latseq, fringe=False)
env_solver_nodes = set_env_solver_nodes(lattice, init_env.perveance, max_solver_spacing)
env_monitor_nodes = [add_analysis_node(lattice, ws_name, 'env_monitor') for ws_name in ws_names]
    
    
# Perform scan
#------------------------------------------------------------------------------
intensities = np.linspace(0, 10.0e14, 20)
transfer_mats_list, moments_list, emittances_list = [], [], []
for intensity in tqdm(intensities):
    transfer_mats, moments = [], []
    for quad_strengths in quad_strengths_list:
        controller.set_quad_strengths(quad_strengths)
        controller.apply_settings(lattice)
        for monitor_node in env_monitor_nodes:
            monitor_node.clear_data()
        env = init_env.copy()
        env.set_intensity(intensity)
        set_perveance(env_solver_nodes, env.perveance)
        env.track(lattice)
        for ws_name, monitor_node in zip(ws_names, env_monitor_nodes):
            env.params = monitor_node.get_data('env_params')
            Sigma = env.cov()
            moments.append([Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]])
            transfer_mats.append(controller.get_transfer_matrix(ws_name))
    Sigma = reconstruct(transfer_mats, moments)
    Sigma *= 1e6 # convert to mm mrad
    eps1, eps2 = intrinsic_emittances(Sigma)
    print(eps1, eps2)
    epsx, epsy = apparent_emittances(Sigma)
    emittances_list.append([eps1, eps2, epsx, epsy])
    moments_list.append(moments)
    transfer_mats_list.append(transfer_mats)

np.save('_output/data/emittances_list.npy', emittances_list)
np.save('_output/data/moments_list.npy', moments_list)
np.save('_output/data/transfer_mats_list.npy', transfer_mats_list)
np.save('_output/data/intensities.npy', intensities)