"""Compare the s-dependent beam envelope parameters for two beams.

Currently three simulation types are available:
    (0) Mismatched beam vs. matched beam. In the former case a beam is created
        which is matched to the bare lattice, but tracked with the inclusion of
        space charge. In the latter case the beam is matched to the lattice with space
        charge.
    (1) Mode 1 vs. mode 2. The modes differ in which of the beam's two intrinsic
        emittances is set to zero.
    (2) Bare lattice vs. lattice with space charge. Both beams are matched to the bare
        lattice, but the second is tracked with the inclusion of space charge.
        
This should be reworked to include turn-by-turn tracking and multi-particle tracking. 
It won't take too long to do so.
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
sim_type = 0 # see options in comments at top of script
n_parts = int(1e5)
n_test_parts = 100
n_turns_track = 10
tracking = 'turn_by_turn' # {'turn_by_turn', 'within_lattice'}
track_bunch = True
store_bunch_coords = True
dense = True  

# Lattice
# madx_file = '_input/SNSring_nux6.18_nuy6.18.lat'
# madx_seq = 'rnginj'
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
    'transfer_matrix': '_output/data/transfer_matrix.dat',
    'xvals': '_output/data/xvals.dat' # turn number or position in lattice
}
delete_files_not_folders('./_output/')


# Initialize
#------------------------------------------------------------------------------
print('Simulation type: {}'.format(sim_type))

lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)
perveance = hf.get_perveance(mass, kin_energy, intensity / bunch_length)
env_solver_nodes = set_env_solver_nodes(lattice, perveance, max_solver_spacing)

def initialize(mode=1, match=True, I=0):
    """Create envelope matched to the bare lattice, then (possibly) match 
    with space charge."""
    env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, lattice.getLength())
    env.match_bare(lattice, solver_nodes=env_solver_nodes)
    env.set_intensity(intensity)
    if match:
        print('Matching.')
        env.match(lattice, env_solver_nodes, verbose=2)
    return env

# Initialize the two envelopes
if sim_type == 0:
    envelopes = [initialize(mode, _match) for _match in (False, True)]
elif sim_type == 1:
    envelopes = [initialize(_mode, match=True) for _mode in (1, 2)]
elif sim_type == 2:
    envelopes = [initialize(mode, match=False) for _ in (1, 2)]
else:
    print('Unknown sim_type.')
    
# Store the column titles for two-column comparison figures
file = open('_output/figures/figure_column_titles.txt', 'w')
if sim_type == 0:
    file.write('Unmatched/Matched')
elif sim_type == 1:
    file.write('Mode 1/Mode 2')
elif sim_type == 2:
    file.write('Q = 0/Q > 0')
else:
    file.write('Left/Right')
    
# Store the beam mode for each column
modes = [1, 2] if sim_type == 1 else [mode, mode]
np.savetxt('_output/data/modes.txt', modes)
    
    
# Simulation
#------------------------------------------------------------------------------
env_monitor_nodes = add_analysis_nodes(DanilovEnvelopeBunchMonitorNode, 
                                       lattice, dense=dense)
positions = [node.position for node in env_monitor_nodes]
np.save('_output/data/positions.npy', positions)
env_params_list, transfer_matrices = [], []

for i, env in enumerate(envelopes, start=1):
    if sim_type == 2:
        hf.toggle_spacecharge_nodes(env_solver_nodes, 'off' if i == 1 else 'on')
    M = env.transfer_matrix(lattice)
    env.track(lattice)
    env_params = [node.get_data(-1).env_params for node in env_monitor_nodes]
    for node in env_monitor_nodes:
        node.clear_data()
    np.save('_output/data/transfer_matrix{}.npy'.format(i), M)
    np.save('_output/data/env_params{}.npy'.format(i), env_params)
    print('Simulation {} complete.'.format(i))