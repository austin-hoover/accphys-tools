"""
This script repeatedly tracks a beam through the RTBT, each time varying the 
optics and computing the phase advance, transfer matrix, and beam moments at 
the 5 wire-scanners.
"""

# Standard 
import sys
import time
import copy
# Third party
import numpy as np
from numpy import pi
import pandas as pd
from scipy import optimize as opt
from tqdm import tqdm
# PyORBIT
from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode, WireScannerNode, add_analysis_node, add_ws_node
from orbit.envelope import Envelope
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

sys.path.append('/Users/46h/Research/code/accphys/pyorbit_studies/measurement')
from utils import (PhaseController, set_rtbt_quad_strengths)


# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # GeV/c^2
kin_energy = 0.8 # GeV
intensity = 0.0e14
bunch_length = 150.0 # [m]
nparts = int(1e4)
max_solver_spacing = 0.02
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)
latfile = '_input/rtbt.lat'
latseq = 'surv'

# Initial beam
beam_type = 'danilov'
eps = 20e-6 # nonzero intrinsice emittance ex + ey [m*rad]
mode = 1
ex_frac = 0.5 
ex, ey = ex_frac * eps, (1 - ex_frac) * eps
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)

# Scan
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
nsteps = 15 # number of steps for each dimension
wsbins = 50
phase_coverage = 180 # [deg]
max_betas = (40, 40) # (x, y)
diag_wire_angle = np.radians(45.0)


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

# Save default optics
latt = hf.lattice_from_file(latfile, latseq)
controller = PhaseController(latt, init_twiss, mass, kin_energy, ref_ws_name)
twiss_df = pd.DataFrame(np.copy(controller.tracked_twiss),
                        columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('_output/data/twiss.dat', index=False)

# Save wire-scanner positions
ws_positions = [controller.get_node_position(ws) for ws in ws_names]
np.savetxt('_output/data/ws_positions.dat', ws_positions)

    
# Set up initial beam
#------------------------------------------------------------------------------
ax0, ay0, bx0, by0 = init_twiss
env = Envelope(eps, mode, ex_frac, mass, kin_energy, bunch_length, intensity)
env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)
env_params0 = np.copy(env.params)

if beam_type == 'danilov':
    X0 = env.generate_dist(nparts, mm_mrad=False)
elif beam_type == 'gaussian':
    cut_off = 3
    bunch, params_dict = hf.coasting_beam(
        beam_type, nparts, init_twiss, (ex, ey), bunch_length, mass, 
        kin_energy, intensity, cut_off=cut_off)
    X0 = hf.dist_from_bunch(bunch)

def reset_bunch():
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    hf.dist_to_bunch(X0, bunch, bunch_length)
    bunch.macroSize(intensity / nparts)
    return bunch, params_dict

np.save('_output/data/Sigma0_env.npy', env.cov())
np.savetxt('_output/data/Sigma0.dat', np.cov(X0.T))
np.savetxt('_output/data/X0.dat', X0)


# Create lattice
#------------------------------------------------------------------------------
lattice = hf.lattice_from_file(latfile, latseq)

# Add space charge nodes
if intensity > 0:
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# Add wire-scanner nodes
ws_nodes = {ws_name: add_ws_node(lattice, ws_name, wsbins, diag_wire_angle)
            for ws_name in ws_names}

# Add bunch monitor nodes
bunch_monitor_nodes = {ws_name: add_analysis_node(lattice, ws_name, 'bunch_monitor')
                       for ws_name in ws_names}

# Create separate lattice for envelope tracking
env_lattice = hf.lattice_from_file(latfile, latseq)
set_env_solver_nodes(env_lattice, env.perveance, max_solver_spacing)
env_monitor_nodes = {ws_name: add_analysis_node(env_lattice, ws_name, 'env_monitor')
                     for ws_name in ws_names}


# Perform scan
#------------------------------------------------------------------------------
def init_dict():
    return {name:[] for name in ws_names}

transfer_mats = init_dict()
moments = init_dict()
coords = init_dict()
phases = init_dict()
env_params = init_dict()

window = 0.5 * phase_coverage
delta_nu_list = np.linspace(-window, window, nsteps) / 360
nux0, nuy0 = controller.get_phases_at_ref_ws()

for direction in ('horizontal', 'vertical'):
    print 'Scanning {} phases.'.format(direction)
    
    for delta_nu in tqdm(delta_nu_list):
        if direction == 'horizontal':
            nux, nuy = nux0 + delta_nu, nuy0
        elif direction == 'vertical':
            nux, nuy = nux0, nuy0 + delta_nu
        
        # Set phases at reference wire-scanner
        print ' delta_nu = {:.3f} deg'.format(360 * delta_nu)
        print 'Setting phases at {}.'.format(ref_ws_name)
        controller.set_phases_at_ref_ws(nux, nuy, max_betas, verbose=2)
        controller.apply_settings(lattice)
        
        # Compute phases and transfer matrix at each wire-scanner
        controller.track_twiss()
        for ws in ws_names:
            phases[ws].append(controller.get_phases(ws))
            transfer_mats[ws].append(controller.get_transfer_matrix(ws))
                    
        # Track bunch and compute moments at each wire-scanner
        for node in bunch_monitor_nodes.values():
            node.clear_data()
        print '  Tracking bunch.'
        bunch, params_dict = reset_bunch()
        lattice.trackBunch(bunch, params_dict)
        for ws in ws_names:
            moments[ws].append(ws_nodes[ws].get_moments())
            coords[ws].append(bunch_monitor_nodes[ws].get_data('bunch_coords'))
            
        # Track envelope for comparison
        controller.apply_settings(env_lattice)
        for node in env_monitor_nodes.values():
            node.clear_data()
        env.params = env_params0
        env.track(env_lattice)
        for ws in ws_names:
            env_params[ws].append(env_monitor_nodes[ws].get_data('env_params'))
            
            
# Save data
#------------------------------------------------------------------------------
for ws in ws_names:
    np.save('_output/data/phases_{}.npy'.format(ws), phases[ws])
    np.save('_output/data/transfer_mats_{}.npy'.format(ws), transfer_mats[ws])
    np.save('_output/data/moments_{}.npy'.format(ws), moments[ws])
    
    np.save('_output/data/env_params_{}.npy'.format(ws), env_params[ws])
    np.save('_output/data/X_{}.npy'.format(ws), coords[ws])