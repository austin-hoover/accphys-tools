"""
Store the matched beam parameters as a function of s for each (lattice, mode, 
intensity). 
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
intensities = np.linspace(0, 10e14, 8)

# Lattice
latfiles = [
    '_latfiles/fodo_quadstart.lat',
    '_latfiles/fodo_split_quadstart.lat',
    '_latfiles/fodo_skew_quadstart.lat',
    '_latfiles/fodo_sol_quadstart.lat',
]
latnames = [
    'fodo',
    'fodo_split',
    'fodo_skew',
    'fodo_sol'
]
latseq = 'fodo'
fringe = False

# Initial beam
eps = 50e-6 # intrinsic emitance
ex_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Matching
match = True 
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}

# Space charge solver
max_solver_spacing = 0.01 # [m]

# Output data locations
files = {
    'positions': '_output/data/positions_{}.npy', 
    'perveances': '_output/data/perveances.npy', 
    'tracked_env_params_list': '_output/data/tracked_env_params_list_{}_{}.npy',
    'transfer_mats': '_output/data/transfer_mats_{}_{}.npy',
}
data = {key:None for key in files.keys()}

delete_files_not_folders('_output')

#------------------------------------------------------------------------------

for latname, latfile in zip(latnames, latfiles):
    print latfile
    
    for mode in (1, 2):
        print 'Mode', mode
        lattice = hf.lattice_from_file(latfile, latseq, fringe)
        env = Envelope(eps, mode, ex_frac, mass, energy, lattice.getLength())
        tracked_env_params_list, transfer_mats = [], []

        for intensity in tqdm(intensities):

            lattice = hf.lattice_from_file(latfile, latseq, fringe)
            env.match_bare(lattice)
            env.set_spacecharge(intensity)
            solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
            if match and intensity > 0:
                env.match(lattice, solver_nodes, verbose=0)
                
            # For intensity == 0 in the skew quad lattice, we can end up with 
            # a beam with zero area. It leads to 
            

            transfer_mats.append(env.transfer_matrix(lattice))
            env_monitor_nodes = add_analysis_nodes(lattice, kind='env_monitor')
            env.track(lattice)
            tracked_env_params = get_analysis_nodes_data(env_monitor_nodes, 'env_params')
            tracked_env_params_list.append(tracked_env_params)

            # Save data
            data['positions'] = get_analysis_nodes_data(env_monitor_nodes, 'position')
            data['perveances'] = hf.get_perveance(mass, energy, intensities/lattice.getLength())
            data['tracked_env_params_list'] = tracked_env_params_list
            data['transfer_mats'] = transfer_mats

            for key in files.keys():
                np.save(files[key].format(latname, mode), data[key])
            
            
file = open('_output/data/latnames.txt', 'w')
for latname in latnames:
    file.write(latname + '\n')