"""
This script compares the s-dependent beam envelope parameters for two beams in the same
lattice. Currently two simulation types are available:
    1. Mismatched beam vs. matched beam. In the former case a beam is created
       which is matched to the bare lattice, but tracked with the inclusion of
       space charge. In the latter case the beam is matched to the lattice with space
       charge.
    2. Mode 1 vs. mode 2. The modes differ in which of the beam's two intrinsic
       emittances is set to zero.
"""

# Standard 
import sys
# Third party
import numpy as np
from tqdm import tqdm, trange
# PyORBIT
from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import (
    AnalysisNode, 
    add_analysis_nodes, 
    get_analysis_nodes_data, 
    clear_analysis_nodes_data)
from orbit.envelope import Envelope
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys') 
from tools.utils import delete_files_not_folders
    
    
# Settings
#------------------------------------------------------------------------------

# Simulation type
sim_type = ['unmatched_vs_matched', 'mode1_vs_mode2', 'other'][0]

# General
mass = 0.93827231 # GeV/c^2
energy = 1.0 # GeV/c^2
intensity = 4e14

# Lattice
latfile = '_latfiles/fodo_quadstart.lat'
latseq = 'fodo'
fringe = False

# Initial beam
mode = 2
eps = 50e-6 # intrinsic emitance
ex_frac = 0.5 # ex/eps
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

print 'Simulation type:', sim_type
delete_files_not_folders('_output/')

lattice = hf.lattice_from_file(latfile, latseq, fringe)
perveance = hf.get_perveance(mass, energy, intensity/lattice.getLength())
solver_nodes = set_env_solver_nodes(lattice, perveance, max_solver_spacing)

def initialize(mode=1, match=True):
    env = Envelope(eps, mode, ex_frac, mass, energy, lattice.getLength())
    env.match_bare(lattice, sc_nodes=solver_nodes)
    env.set_spacecharge(intensity)
    if match:
        print 'Matching.'
        env.match(lattice, solver_nodes, verbose=2)
    return env

# Initialize the two envelopes
if sim_type == 'unmatched_vs_matched':
    envelopes = [initialize(mode, _match) for _match in (False, True)]
elif sim_type == 'mode1_vs_mode2':
    envelopes = [initialize(_mode) for _mode in (1, 2)]
else:
    print 'Unknown sim_type.'
    envelopes = [initialize() for _ in range(2)]
    
# Store the column titles for two-column comparison figures
file = open('_output/figures/figure_column_titles.txt', 'w')
if sim_type == 'unmatched_vs_matched':
    file.write('Unmatched/Matched')
elif sim_type == 'mode1_vs_mode2':
    file.write('Mode 1/Mode 2')
else:
    file.write('Left/Right')
    
# Store the beam mode for each column
modes = [1, 2] if sim_type == 'mode1_vs_mode2' else [mode, mode]
np.savetxt('_output/data/modes.txt', modes)

    
# Simulation
#------------------------------------------------------------------------------

env_monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
positions = get_analysis_nodes_data(env_monitor_nodes, 'position')
np.save('_output/data/positions.npy', positions)
env_params_list, transfer_matrices = [], []

for i, env in enumerate(envelopes, start=1):
    M = env.transfer_matrix(lattice)
    env.track(lattice)
    env_params = get_analysis_nodes_data(env_monitor_nodes, 'env_params')
    clear_analysis_nodes_data(env_monitor_nodes)
    np.save('_output/data/transfer_matrix{}.npy'.format(i), M)
    np.save('_output/data/env_params{}.npy'.format(i), env_params)
    print 'Iteration {} complete.'.format(i)