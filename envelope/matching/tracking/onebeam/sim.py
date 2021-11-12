"""
This script calculates the matched envelope of the Danilov distribution
and tracks it through the lattice.

Options:
* Store the turn-by-turn or s-dependent beam parameters.
* Track test particles in the field of the envelope + lattice.
* Track a distribution generated from the envelope.
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
n_turns_track = 20
tracking = 'turn_by_turn' # {'turn_by_turn', 'within_lattice'}
track_bunch = True
store_bunch_coords = True
# If tracking the s-dependent beam parameters, we need to choose where to 
# put the monitor nodes. Normally, a monitor node is placed as a child of 
# every lattice node. If `dense` is True, a monitor node is placed at every
# part of every node. In this script, this means that a monitor node is
# placed at every envelope solver node.
dense = True  

# Lattice
# madx_file = '_input/fodo_driftstart.lat'
# madx_seq = 'fodo'
madx_file = '_input/SNSring_nux6.18_nuy6.18.lat'
madx_seq = 'rnginj'
fringe = False
print('madx_file = {}'.format(madx_file))

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
perturb_radius = 0. # between 0 (no effect) and 1.0
method = 'auto' # {'lsq', 'replace_by_avg', 'auto'}

# Output data locations
filenames = {
    'env_params': '_output/data/envelope/env_params.dat',
    'test_bunch_coords': '_output/data/envelope/test_bunch_coords.npy',
    'bunch_coords': '_output/data/bunch/bunch_coords.npy',
    'bunch_moments': '_output/data/bunch/bunch_moments.dat',
    'transfer_matrix': '_output/data/transfer_matrix.dat',
    'xvals': '_output/data/xvals.dat' # turn number or position in lattice
}
delete_files_not_folders('./_output/')
    
    
# Envelope
#------------------------------------------------------------------------------
# Create an envelope matched to the bare lattice using the transfer matrix
# eigenvectors. If the lattice is uncoupled, the beam will be flat (zero 
# emittance in x-x' or y-y'). If the tunes are equal, any vector is an 
# eigenvector and the matching can just done in x-x' and y-y', avoiding the
# flat beam. This is done by calling `env.match_bare(lattice, '2D')`.
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)
env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
env.match_bare(lattice, '2D')
    
# Match the envelope to the lattice with space charge. The least squares 
# will always terminate, but it is possible that it may not find the matched 
# solution. The 'replace_by_avg' method is not guaranteed to converge.
env.set_intensity(intensity)
env_solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
if intensity == 0.:
    for env_solver_node in env_solver_nodes:
        env_solver_node.setCalculationOn(False)
if match and intensity > 0:
    print('Matching.')
    env.match(lattice, env_solver_nodes, tol=tol, verbose=verbose, method=method)
    
# Optionally perturb the envelope around the matched solution.
if perturb_radius > 0:
    print('Perturbing envelope (radius = {:.2f}).'.format(perturb_radius))
    env.perturb(perturb_radius)
init_env_params = np.copy(env.params)
    
# Compute the linear transfer matrix of the 'effective lattice' 
# (lattice + linear space charge).
M = env.transfer_matrix(lattice)
tune_x, tune_y = env.tunes(lattice)
phase_adv_x, phase_adv_y = 360. * np.array([tune_x, tune_y])
if twiss.is_stable(M):
    print('Effective transfer matrix is stable.')
else:
    print('Transfer matrix is unstable.')
print('    phase_adv_x, phase_adv_y = {:.3f}, {:.3f} deg'.format(phase_adv_x, phase_adv_y))
print('    tune_x, tune_y = {:.3f}, {:.3f}'.format(tune_x, tune_y))

# Add envelope monitor node(s). Each node stores a list of data that is
# appended to every time it tracks the bunch. 
print('Tracking envelope.')
if tracking == 'turn_by_turn':
    first_node = lattice.getNodes()[0]
    env_monitor_node = add_analysis_node(
        DanilovEnvelopeBunchMonitorNode, lattice, first_node
    )
elif tracking == 'within_lattice':
    env_monitor_nodes = add_analysis_nodes(
        DanilovEnvelopeBunchMonitorNode, lattice, dense=dense,
    )
    
# Track the envelope.
env.track(lattice, n_turns_track, n_test_parts, progbar=True)

# Extract the envelope parameters, test particle coordinates, and 
# node positions from the envelope monitor nodes.
print('Saving envelope data.')
if tracking == 'turn_by_turn':
    env_data = env_monitor_node.get_data()
    xvals = np.arange(n_turns_track)
elif tracking == 'within_lattice':
    positions = np.array([node.position for node in env_monitor_nodes])
    env_data, xvals = [], []
    for turn in range(n_turns_track):
        env_data.extend([node.get_data(turn) for node in env_monitor_nodes])
        offset = turn * lattice.getLength()
        xvals.extend(offset + positions)
else:
    raise ValueError("`tracking` is invalid")

env_params = [d.env_params for d in env_data]
test_bunch_coords = [d.coords for d in env_data]

# Save all envelope data.
np.savetxt(filenames['env_params'], env_params)
np.save(filenames['test_bunch_coords'], test_bunch_coords)
np.savetxt(filenames['transfer_matrix'], M)
np.savetxt(filenames['xvals'], xvals)

file = open('_output/data/xlabel.txt', 'w')
if tracking == 'turn_by_turn':
    file.write('Turn number')
elif tracking == 'within_lattice':
    file.write('Position [m]')
else:
    raise ValueError("`tracking` is invalid")
file.close()

file = open('_output/data/mode.txt', 'w')
file.write('{}'.format(mode))
file.close()


# Bunch
#------------------------------------------------------------------------------
if not track_bunch:
    exit()
    
# Create a new lattice with 2.5D space charge nodes.
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(fringe)
lattice.split(max_solver_spacing)    
calc2p5d = SpaceChargeCalc2p5D(*gridpts)
sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
if intensity == 0.:
    for node in sc_nodes:
        node.setCalculationOn(False)

# Generate a distribution from the envelope. Note that there is the 
# option to vary the radial density while maintaining the same covariance
# matrix.
env.params = init_env_params
bunch, params_dict = env.to_bunch(n_parts, no_env=True)

# Add bunch monitor node(s). 
if tracking == 'turn_by_turn':
    first_node = lattice.getNodes()[0]
    if store_bunch_coords:
        bunch_monitor_node = add_analysis_node(BunchMonitorNode, lattice, first_node)   
    bunch_stats_node = add_analysis_node(BunchStatsNode, lattice, first_node)  
elif tracking == 'within_lattice':
    if store_bunch_coords:
        bunch_monitor_nodes = add_analysis_nodes(BunchMonitorNode, lattice, dense)
    bunch_stats_nodes = add_analysis_nodes(BunchStatsNode, lattice, dense=dense)  
else:
    raise ValueError("`tracking` is invalid")

# Track the bunch.
print('Tracking bunch.')
for turn in trange(n_turns_track):
    lattice.trackBunch(bunch, params_dict)
    
# Extract bunch moments and (maybe) the full 6D phase space coordinates
# from each node. The node positions should be the same as for the 
# envelope solver.
print('Saving bunch data.')
if tracking == 'turn_by_turn':
    moments = bunch_stats_node.get_data()
    if store_bunch_coords:
        coords = bunch_monitor_node.get_data()
elif tracking == 'within_lattice':
    moments = []
    for turn in range(n_turns_track):
        moments.extend([node.get_data(turn).moments for node in bunch_stats_nodes])
    if store_bunch_coords:
        coords = []
        for turn in range(n_turns_track):
            coords.extend([node.get_data(turn) for node in bunch_monitor_nodes])
else:
    raise ValueError("`tracking` is invalid")

# Save all bunch data.
np.savetxt(filenames['bunch_moments'], moments)
if store_bunch_coords:
    np.save(filenames['bunch_coords'], coords)