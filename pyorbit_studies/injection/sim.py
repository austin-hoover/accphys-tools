"""
This script is to test injection routines ported from Jeff Holmes' local 
copy of PyORBIT.
"""

# Standard 
import sys
# Third party
import numpy as np
from tqdm import tqdm, trange
# PyORBIT
from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode
from orbit.coupling import bogacz_lebedev as BL
from orbit.envelope import Envelope
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys') 
from tools.utils import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------

# General
mass = 0.93827231 # GeV/c^2
energy = 1.0 # GeV/c^2
intensity = 1e14
nturns_track = 30
nparts = int(1e5)
ntestparts = 100
track_bunch = True
store_bunch_coords = True

# Lattice
latfile = '_latfiles/SNSring_linear_noRF_nux6.18_nuy6.18.lat'
latseq = 'fodo'
fringe = False

# Initial beam
mode = 1
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
perturb_radius = 0.2 # If nonzero, perturb the matched envelope
method = 'auto' # 'lsq' or 'replace_by_avg'

# Output data locations
files = {
    'env_params': '_output/data/envelope/env_params.npy',
    'testbunch_coords': '_output/data/envelope/testbunch_coords.npy',
    'bunch_coords': '_output/data/bunch/bunch_coords.npy',
    'bunch_moments': '_output/data/bunch/bunch_moments.npy',
    'transfer_matrix': '_output/data/transfer_matrix.npy'
}

delete_files_not_folders('_output/')

        
# Envelope
#------------------------------------------------------------------------------

# Create envelope matched to bare lattice
lattice = hf.lattice_from_file(latfile, latseq, fringe)




# Bunch
#------------------------------------------------------------------------------
if track_bunch:
    
    # Create lattice with space charge nodes
    lattice = hf.lattice_from_file(latfile, latseq, fringe)
    lattice.split(max_solver_spacing)    
    if intensity > 0:
        calc2p5d = SpaceChargeCalc2p5D(*gridpts)
        sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

    # Create bunch
    env.params = init_params
    bunch, params_dict = env.to_bunch(nparts, no_env=True)

    # Add analysis nodes
    bunch_monitor_node = AnalysisNode(0.0, 'bunch_monitor')
    bunch_stats_node = AnalysisNode(0.0, 'bunch_stats')
    hf.add_node_at_start(lattice, bunch_stats_node)
    if store_bunch_coords:
        hf.add_node_at_start(lattice, bunch_monitor_node)
    
    # Track bunch
    print 'Tracking bunch.'
    hf.track_bunch(bunch, params_dict, lattice, nturns_track)
    
    # Save data
    moments = bunch_stats_node.get_data('bunch_moments', 'all_turns')
    np.save(files['bunch_moments'], moments)
    if store_bunch_coords:
        coords = bunch_monitor_node.get_data('bunch_coords', 'all_turns')
        np.save(files['bunch_coords'], coords)