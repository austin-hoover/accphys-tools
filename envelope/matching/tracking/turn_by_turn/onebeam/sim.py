"""
This script calculates the matched envelope of the Danilov distribution, then
tracks the envelope and/or bunch over a number of turns. It will save the
turn-by-turn envelope parameters, bunch moments, and/or bunch coordinates.

The saved file formats are '.npy', which is convenient for storing multi-dim
arrays. They can be loaded by calling `np.load(filename)`.
"""
from __future__ import print_function
import sys
import os

import numpy as np
from tqdm import tqdm
from tqdm import trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import DanilovEnvelopeBunchMonitorNode
from orbit.diagnostics import add_analysis_node
from orbit.diagnostics import add_analysis_nodes
from orbit.envelope import DanilovEnvelope
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.envelope import set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf
from orbit.utils.general import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV/c^2]
intensity = 1.5e14
n_parts = int(1e5)
ntestparts = 100
n_turns_track = 10
track_bunch = True
store_bunch_coords = True

# Lattice
madx_file = '_input/fodo_quadstart.lat'
madx_seq = 'fodo'
fringe = False

# Initial beam
bunch_length = 150.0 # [m]
mode = 2
eps_l = 40e-6 # nonzero intrinsic emittance [m rad]
eps_x_frac = 0.5 # eps_x / eps_l
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.1 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
match = True 
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0.0 # between 0 (no effect) and 1.0
method = 'auto' # {'lsq', 'replace_by_avg'}

# Output data locations
files = {
    'env_params': '_output/data/envelope/env_params.npy',
    'testbunch_coords': '_output/data/envelope/testbunch_coords.npy',
    'bunch_coords': '_output/data/bunch/bunch_coords.npy',
    'bunch_moments': '_output/data/bunch/bunch_moments.npy',
    'transfer_matrix': '_output/data/transfer_matrix.npy'
}

delete_files_not_folders('./_output/')
    
        
# Envelope
#------------------------------------------------------------------------------

# Create envelope matched to bare lattice
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(fringe)
env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
env.match_bare(lattice, '2D') # if '4D', and unequal tunes, beam will have 0 area
    
# Create envelope 
env.set_intensity(intensity)
solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
if match and intensity > 0:
    print('Matching.')
    env.match(lattice, solver_nodes, tol=tol, verbose=verbose, method=method)
if perturb_radius > 0:
    print('Perturbing envelope (radius = {:.2f}).'.format(perturb_radius))
    env.perturb(perturb_radius)
init_params = np.copy(env.params)
    
# Get linear transfer matrix
M = env.transfer_matrix(lattice)
mux, muy = 360 * env.tunes(lattice)
print('Transfer matrix is {}stable.'.format('' if twiss.is_stable(M) else 'un'))
print('    mux, muy = {:.3f}, {:.3f} deg'.format(mux, muy))

# Add envelope monitor node.
first_node = lattice.getNodes()[0]
env_monitor_node = add_analysis_node(DanilovEnvelopeBunchMonitorNode, lattice, first_node)

# Track envelope
print('Tracking envelope.')
env.track(lattice, n_turns_track, ntestparts, progbar=True)

# Save data
env_data_tbt = env_monitor_node.get_data()
env_params_tbt = [env_data.env_params for env_data in env_data_tbt]
testbunch_coords = [env_data.coords for env_data in env_data_tbt]
np.save(files['env_params'], env_params_tbt)
np.save(files['testbunch_coords'], testbunch_coords)
np.save(files['transfer_matrix'], M)
np.savetxt('_output/data/mode.txt', [mode])


# Bunch
#------------------------------------------------------------------------------
if track_bunch:
    
    # Create lattice with space charge nodes.
    lattice = TEAPOT_Lattice()
    lattice.readMADX(madx_file, madx_seq)
    lattice.set_fringe(fringe)
    lattice.split(max_solver_spacing)    
    if intensity > 0:
        calc2p5d = SpaceChargeCalc2p5D(*gridpts)
        sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

    # Create bunch.
    env.params = init_params
    bunch, params_dict = env.to_bunch(n_parts, no_env=True)

    # Add analysis nodes.
    first_node = lattice.getNodes()[0]
    if store_bunch_coords:
        bunch_monitor_node = add_analysis_node(BunchMonitorNode, lattice, first_node)   
    bunch_stats_node = add_analysis_node(BunchStatsNode, lattice, first_node)   
    
    # Track bunch.
    print('Tracking bunch.')
    hf.track_bunch(bunch, params_dict, lattice, n_turns_track)
    
    # Save data.
    bunch_moments = bunch_stats_node.get_data()
    np.save(files['bunch_moments'], bunch_moments)
    if store_bunch_coords:
        bunch_coords = bunch_monitor_node.get_data()
        np.save(files['bunch_coords'], bunch_coords)