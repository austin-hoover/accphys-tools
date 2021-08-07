"""
This script compares the s-dependent beam envelope parameters for two beams in the same
lattice. Currently three simulation types are available:
    (0) Mismatched beam vs. matched beam. In the former case a beam is created
        which is matched to the bare lattice, but tracked with the inclusion of
        space charge. In the latter case the beam is matched to the lattice with space
        charge.
    (1) Mode 1 vs. mode 2. The modes differ in which of the beam's two intrinsic
        emittances is set to zero.
    (2) Bare lattice vs. lattice with space charge. Both beams are matched to the bare
        lattice, but the second is tracked with the inclusion of space charge.
"""
import sys
import numpy as np
from tqdm import tqdm, trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode
from orbit.analysis import add_analysis_nodes
from orbit.analysis import get_analysis_nodes_data
from orbit.envelope import DanilovEnvelope
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf

sys.path.append('/Users/46h/Research/code/accphys') 
from tools.utils import delete_files_not_folders
    
    
# Settings
#------------------------------------------------------------------------------
sim_type = 0 # see options in comments at top of script

# General
mass = 0.93827231 # GeV/c^2
kin_energy = 1.0 # GeV/c^2
intensity = 4e14
bunch_length = 150.0 # m

# Lattice
latfile = '_latfiles/fodo_driftstart.lat'
latseq = 'fodo'
fringe = False

# Initial beam
mode = 1
eps = 40e-6 # nonzero intrinsic emittance
eps_x_frac = 0.3 # eps_x / eps
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.05 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
match = True 
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}


# Initialize
#------------------------------------------------------------------------------
print 'Simulation type: {}'.format(sim_type)
delete_files_not_folders('_output/')

lattice = hf.lattice_from_file(latfile, latseq, fringe)
perveance = hf.get_perveance(mass, kin_energy, intensity / bunch_length)
solver_nodes = set_env_solver_nodes(lattice, perveance, max_solver_spacing)

def initialize(mode=1, match=True, I=0):
    """Create envelope matched to the bare lattice, then (possibly) match 
    with space charge."""
    env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, lattice.getLength())
    env.match_bare(lattice, sc_nodes=solver_nodes)
    env.set_intensity(intensity)
    if match:
        print 'Matching.'
        env.match(lattice, solver_nodes, verbose=2)
    return env

# Initialize the two envelopes
if sim_type == 0:
    envelopes = [initialize(mode, _match) for _match in (False, True)]
elif sim_type == 1:
    envelopes = [initialize(_mode, match=True) for _mode in (1, 2)]
elif sim_type == 2:
    envelopes = [initialize(mode, match=False) for _ in (1, 2)]
else:
    print 'Unknown sim_type.'
    
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
env_monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
positions = get_analysis_nodes_data(env_monitor_nodes, 'position')
np.save('_output/data/positions.npy', positions)
env_params_list, transfer_matrices = [], []

for i, env in enumerate(envelopes, start=1):
    if sim_type == 2:
        hf.toggle_spacecharge_nodes(solver_nodes, 'off' if i == 1 else 'on')
    M = env.transfer_matrix(lattice)
    env.track(lattice)
    env_params = get_analysis_nodes_data(env_monitor_nodes, 'env_params')
    for monitor_node in env_monitor_nodes:
        monitor_node.clear_data()
    np.save('_output/data/transfer_matrix{}.npy'.format(i), M)
    np.save('_output/data/env_params{}.npy'.format(i), env_params)
    print 'Iteration {} complete.'.format(i)