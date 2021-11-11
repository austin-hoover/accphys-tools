"""
This script examines the matched solutions in a FODO lattice as the 
horizontal and vertical tunes are split.
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
mass = 0.93827231 # GeV/c^2
kin_energy = 1.0 # GeV
intensity = 3e14
bunch_length = 5.0 # m
match = True

# Lattice
mux = 80 # total x phase advance [deg]
muy_list = [70, 75, 80, 85, 90] # total y phase advance [deg]
length = 5.0 # lattice length [m]
fill_fac = 0.5 # fraction of lattice occupied by quadrupoles
angle = 0.0 # tilt angle of quadrupoles [deg]
start = 'quad' # lattice entrance (the center of quad vs. center of drift)
fringe = False # whether to include fringe fields

# Initial beam
mode = 2
eps = 40e-6 # intrinsic emitance
eps_x_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Matching
match = True
verbose = 2 
max_solver_spacing = 0.01 # [m]

# Output data locations
files = {
    'positions': '_output/data/positions.npy', 
    'phase_adv_diffs': '_output/data/phase_adv_diffs.npy', 
    'tracked_params_list': '_output/data/tracked_params_list.npy',
    'transfer_mats': '_output/data/transfer_mats.npy',
}
data = {key:None for key in files.keys()}

delete_files_not_folders('_output')


# Simulation
#------------------------------------------------------------------------------
env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, bunch_length)
tracked_params_list, transfer_mats = [], []
for muy in tqdm(muy_list):
    lattice = hf.fodo_lattice(mux, muy, length, fill_fac, angle, start)
    env.match_bare(lattice)
    env.set_intensity(intensity)
    solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
    if match and intensity > 0:
        env.match(lattice, solver_nodes, verbose=0)
    transfer_mats.append(env.transfer_matrix(lattice))
    env_monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
    env.track(lattice)
    tracked_params = get_analysis_nodes_data(env_monitor_nodes, 'env_params')
    tracked_params_list.append(tracked_params)

data['positions'] = get_analysis_nodes_data(env_monitor_nodes, 'position')
data['phase_adv_diffs'] = [muy - mux for muy in muy_list]
data['tracked_params_list'] = tracked_params_list
data['transfer_mats'] = transfer_mats

for key in files.keys():
    np.save(files[key], data[key])
    
np.savetxt('_output/data/mode.txt', [mode])
