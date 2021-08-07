"""
This script calculates the matched envelope of the Danilov distribution, then
tracks the envelope and/or bunch over a number of turns. It will save the
turn-by-turn envelope parameters, bunch moments, and/or bunch coordinates.

The saved file formats are '.npy', which is convenient for storing multi-dim
arrays. They can be loaded by calling `np.load(filename)`.
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
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV/c^2]
intensity = 0.0e14
bunch_length = 150.0 # [m]
nparts = int(1e5)
ntestparts = 100
nturns_track = 100
track_bunch = False
store_bunch_coords = False

# Lattice
latfile = '_latfiles/SNSring_from_openxal.lat'
latseq = 'ring_ring'
fringe = False

# Initial beam
mode = 2
eps = 40e-6 # intrinsic emitance
eps_x_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.1 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
match = True 
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0.0 # between 0 (no effect) and 1.0
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
env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
env.match_bare(lattice, '2D') # if '4D', and unequal tunes, beam will have 0 area
    
# Create envelope 
env.set_intensity(intensity)
solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
if match and intensity > 0:
    print 'Matching.'
    env.match(lattice, solver_nodes, tol=tol, verbose=verbose, method=method)
if perturb_radius > 0:
    print 'Perturbing envelope (radius = {:.2f}).'.format(perturb_radius)
    env.perturb(perturb_radius)
init_params = np.copy(env.params)
    
# Get linear transfer matrix
M = env.transfer_matrix(lattice)
mux, muy = 360 * env.tunes(lattice)
print 'Transfer matrix is {}stable.'.format('' if twiss.is_stable(M) else 'un')
print '    mux, muy = {:.3f}, {:.3f} deg'.format(mux, muy)

# Track envelope
print 'Tracking envelope.'
env_monitor_node = AnalysisNode(0.0, 'env_monitor')
hf.add_node_at_start(lattice, env_monitor_node)
env.track(lattice, nturns_track, ntestparts, progbar=True)

# Save data
for key in ('env_params', 'testbunch_coords'):
    np.save(files[key], env_monitor_node.get_data(key, 'all_turns'))
np.save(files['transfer_matrix'], M)
np.savetxt('_output/data/mode.txt', [mode])


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