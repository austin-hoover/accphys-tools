"""
This script first computes the matched envelope using the 'replace by average'
method and stores the beam parameters at each iteration. Then, for each of
these initial beams, it tracks and stores the turn-by-turn and s-dependent
parameters. This allows us to view the progress of the algorithm as the
beam approaches the matched solution. 

Unfortunately there is no `callback` option for scipy.optimize.least_squares,
so we cannot view its progress.
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
# General
intensity = 2e14
mass = 0.93827231 # GeV/c^2
kin_energy = 1.0 # GeV/c^2
bunch_length = 5.0

# Lattice
latfile = '_latfiles/fodo_skew_driftstart.lat'
latseq = 'fodo'
fringe = False

# Initial beam
mode = 1
eps = 50e-6 # intrinsic emitance
eps_x_frac = 0.25 # ex/eps
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.01 # [m]
min_solver_spacing = 1e-6

# Matching
match = True 
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0. # if nonzero, perturb the matched envelope
method = 'replace_avg'

delete_files_not_folders('_output/')


#------------------------------------------------------------------------------

nturns = 15

# Match and store optimizer history
print 'Matching.'
lattice = hf.lattice_from_file(latfile, latseq, fringe)
env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
env.set_intensity(intensity)
solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
result = env.match(lattice, solver_nodes, method=method, verbose=2)

# Using the seed from each iteration, track and store the turn-by-turn 
# envelope parameters at the lattice entrance
print 'Collecting turn-by-turn data.'
tbt_params_list = []
costs = []
for twiss_params in tqdm(result.history):
    env.set_twiss4D(twiss_params)
    costs.append(0.5 * np.sum(env._residuals(lattice)))
    tbt_params = env.track_store_params(lattice, nturns)
    tbt_params_list.append(tbt_params)
np.save('_output/data/tbt_params_list.npy', tbt_params_list)
np.save('_output/data/costs.npy', costs)

# Using the seed from each iteration, track and store the s-dependent
# envelope parameters.
print 'Collecting s-dependent data.'
monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
sdep_params_list = []
for twiss_params in tqdm(result.history):
    env.set_twiss4D(twiss_params)
    env.track(lattice)
    sdep_params = get_analysis_nodes_data(monitor_nodes, 'env_params')
    sdep_params_list.append(sdep_params)
    for monitor_node in monitor_nodes:
        monitor_node.clear_data()
np.save('_output/data/sdep_params_list.npy', sdep_params_list)
np.save('_output/data/positions.npy', 
        get_analysis_nodes_data(monitor_nodes, 'position'))