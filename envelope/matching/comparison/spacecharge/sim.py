"""
This script examines the effect of space charge on the envelope trajectory.
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
bunch_length = 5.0 # m
intensities = np.linspace(0, 10e14, 8)

# Lattice
latfile = '_latfiles/fodo_skew_quadstart.lat'
latseq = 'fodo'
fringe = False

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
    'perveances': '_output/data/perveances.npy', 
    'tracked_env_params_list': '_output/data/tracked_params_list.npy',
    'transfer_mats': '_output/data/transfer_mats.npy',
}
data = {key:None for key in files.keys()}

delete_files_not_folders('_output')


# Simulation
#------------------------------------------------------------------------------
env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, bunch_length)
tracked_params_list, transfer_mats = [], []
for intensity in tqdm(intensities):
    lattice = hf.lattice_from_file(latfile, latseq, fringe)
    env.match_bare(lattice)
    env.set_intensity(intensity)
    solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
    if match and intensity > 0:
        env.match(lattice, solver_nodes, verbose=verbose)
    transfer_mats.append(env.transfer_matrix(lattice))
    env_monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
    env.track(lattice)
    tracked_params = get_analysis_nodes_data(env_monitor_nodes, 'env_params')
    tracked_params_list.append(tracked_params)

# Save data
data['positions'] = get_analysis_nodes_data(env_monitor_nodes, 'position')
data['perveances'] = hf.get_perveance(mass, kin_energy, intensities/lattice.getLength())
data['tracked_env_params_list'] = tracked_params_list
data['transfer_mats'] = transfer_mats
for key in files.keys():
    np.save(files[key], data[key])
np.savetxt('_output/data/mode.txt', [mode])