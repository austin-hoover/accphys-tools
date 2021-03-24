"""
This script examines the matched solutions in a FODO lattice as the tilt angle
of the quadrupoles is varied.
"""

# Standard 
import sys
# Third party
import numpy as np
from tqdm import tqdm
# PyORBIT
from bunch import Bunch
from orbit.analysis import add_analysis_nodes, get_analysis_nodes_data
from orbit.coupling import bogacz_lebedev as BL
from orbit.envelope import Envelope
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------

# General
mass = 0.93827231 # GeV/c^2
energy = 1.0 # GeV
intensity = 3e14

# Lattice
mux = 80 # total x phase advance [deg]
muy = 80 # total y phase advance [deg]
length = 5.0 # lattice length [m]
fill_fac = 0.5 # fraction of lattice occupied by quadrupoles
angles = np.linspace(0, 3, 8) # tilt angle of quadrupoles [deg]
start = 'quad' # lattice entrance (the center of quad vs. center of drift)
fringe = False # whether to include fringe fields

# Initial beam
mode = 2
eps = 50e-6 # intrinsic emitance
ex_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Matching
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}

# Space charge solver
max_solver_spacing = 0.01 # [m]

# Output data locations
files = {
    'positions': '_output/data/positions.npy', 
    'angles': '_output/data/angles.npy', 
    'tracked_env_params_list': '_output/data/tracked_env_params_list.npy',
    'transfer_mats': '_output/data/transfer_mats.npy',
}
data = {key:None for key in files.keys()}

delete_files_not_folders('_output/')

#------------------------------------------------------------------------------

env = Envelope(eps, mode, ex_frac, mass, energy, length)
tracked_env_params_list, transfer_mats = [], []

for angle in angles:
    
    lattice = hf.fodo_lattice(mux, muy, length, fill_fac, angle, start)
    env.match_bare(lattice)
    env.set_spacecharge(intensity)
    solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
    if intensity > 0:
        env.match(lattice, solver_nodes, verbose=2)
    
    transfer_mats.append(env.transfer_matrix(lattice))
    env_monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
    env.track(lattice)
    tracked_env_params = get_analysis_nodes_data(env_monitor_nodes, 'env_params')
    tracked_env_params_list.append(tracked_env_params)

# Save data
data['positions'] = get_analysis_nodes_data(env_monitor_nodes, 'position')
data['angles'] = angles
data['tracked_env_params_list'] = tracked_env_params_list
data['transfer_mats'] = transfer_mats

for key in files.keys():
    np.save(files[key], data[key])
    
np.savetxt('_output/data/mode.txt', [mode])