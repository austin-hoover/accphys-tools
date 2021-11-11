"""
This script calculates the matched envelope of the Danilov distribution, then
tracks the envelope bunch over a number of turns. A real bunch can then be 
generated from the envelope and tracked. The script saves the turn-by-turn 
envelope parameters, bunch moments, and bunch coordinates.
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
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.envelope import set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf
from orbit.utils.consts import mass_proton
from orbit.utils.general import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------
# General
n_parts = int(1e5)
n_test_parts = 100
n_turns_track = 25 
track_bunch = True
store_bunch_coords = True

# Lattice
madx_file = '_input/fodo_driftstart.lat'
madx_seq = 'fodo'
fringe = False

# Initial beam
mass = mass_proton # [GeV/c^2]
kin_energy = 0.8 # [GeV]
intensity = 0.5 * 1.5e14
bunch_length = (45.0 / 64.0) * 248.0 # [m]
mode = 2 # {1, 2} determines sign of angular momentum
eps_l = 20e-6 # nonzero intrinsic emittance [m rad]
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
perturb_radius = 0.1 # between 0 (no effect) and 1.0
method = 'auto' # {'lsq', 'replace_by_avg', 'auto'}

# Output data locations
filenames = {
    'env_params': '_output/data/envelope/env_params.dat',
    'test_bunch_coords': '_output/data/envelope/test_bunch_coords.npy',
    'bunch_coords': '_output/data/bunch/bunch_coords.npy',
    'bunch_moments': '_output/data/bunch/bunch_moments.dat',
    'transfer_matrix': '_output/data/transfer_matrix.dat'
}
delete_files_not_folders('./_output/')
    
        
# Envelope
#------------------------------------------------------------------------------
# Create envelope matched to the bare lattice. Keep in mind that if 4D matching
# is performed and the lattice is uncoupled with unequal tunes, the resulting
# beam will be flat; i.e., have zero emittance in x or y.
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)
env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
env.match_bare(lattice, '2D')
    
# Match the envelope to the lattice with space charge. 
env.set_intensity(intensity)
solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
if match and intensity > 0:
    print('Matching.')
    env.match(lattice, solver_nodes, tol=tol, verbose=verbose, method=method)
    
# Perturb around the matched solution.
if perturb_radius > 0:
    print('Perturbing envelope (radius = {:.2f}).'.format(perturb_radius))
    env.perturb(perturb_radius)
init_env_params = np.copy(env.params)
    
# Compute the linear transfer matrix of the 'effective lattice' (lattice + 
# linear space charge).
M = env.transfer_matrix(lattice)
mux, muy = 360 * env.tunes(lattice)
print('Transfer matrix is {}stable.'.format('' if twiss.is_stable(M) else 'un'))
print('    mux, muy = {:.3f}, {:.3f} deg'.format(mux, muy))

# Add envelope monitor node.
first_node = lattice.getNodes()[0]
env_monitor_node = add_analysis_node(DanilovEnvelopeBunchMonitorNode, lattice, first_node)

# Track envelope.
print('Tracking envelope.')
env.track(lattice, n_turns_track, n_test_parts, progbar=True)

# Save data
env_data_tbt = env_monitor_node.get_data()
env_params_tbt = [env_data.env_params for env_data in env_data_tbt]
test_bunch_coords = [env_data.coords for env_data in env_data_tbt]
np.savetxt(filenames['env_params'], env_params_tbt)
np.save(filenames['test_bunch_coords'], test_bunch_coords)
np.savetxt(filenames['transfer_matrix'], M)
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
    env.params = init_env_params
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
    moments = [stats.moments for stats in bunch_stats_node.get_data()]
    np.savetxt(filenames['bunch_moments'], moments)
    if store_bunch_coords:
        coords = bunch_monitor_node.get_data()
        np.save(filenames['bunch_coords'], coords)