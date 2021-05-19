"""
This script calculates the matched envelope of the Danilov distribution, then
tracks the envelope and/or bunch over a number of turns. It saves the 
turn-by-turn envelope parameters, bunch moments, and/or bunch coordinates.

The saved file formats are '.npy', which is convenient for storing multi-dim
arrays. They can be loaded by calling `np.load(filename)`.
"""

# Standard 
import sys
# Third party
import numpy as np
from tqdm import tqdm, trange
# PyORBIT
from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode, add_analysis_nodes, get_analysis_nodes_data
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
kin_energy = 1.0 # GeV/c^2
intensity = 1.0e14
nturns_track = 50
nparts = 128000
ntestparts = 0
track_bunch = False
store_bunch_coords = False

# Lattice
latfile = '_latfiles/SNSring_linear_noRF_nux6.18_nuy6.18.lat'
latseq = 'rnginj'
fringe = False

# Initial beam
mode = 1
eps = 50e-6 # intrinsic emitance = ex + ey
ex_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.01 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
match = True 
tol = 1e-5 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0.0 # If nonzero, perturb the matched envelope
method = 'lsq' # 'lsq' or 'replace_by_avg'

# Output data locations
files = {
    'env_params': '_output/data/envelope/env_params.npy',
    'testbunch_coords': '_output/data/envelope/testbunch_coords.npy',
    'bunch_coords': '_output/data/bunch/bunch_coords.npy',
    'bunch_moments': '_output/data/bunch/bunch_moments.npy',
    'transfer_matrix': '_output/data/transfer_matrix.npy'
}

delete_files_not_folders('_output/')


def get_skew_quad_nodes(ring, return_names=False):
    skew_quad_nodes, skew_quad_names = [], []
    for node in ring.getNodes():
        name = node.getName()
        if name.startswith('qsc'):
            node.setParam('skews', [0, 1])
            skew_quad_nodes.append(node)
            skew_quad_names.append(name)
    if return_names:
        return skew_quad_nodes, skew_quad_names
    else:
        return skew_quad_nodes
    
    
def set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths):
    for node, strength in zip(skew_quad_nodes, skew_quad_strengths):
        node.setParam('kls', [0.0, strength])
        
ring = hf.lattice_from_file(latfile, latseq, fringe)
skew_quad_nodes, skew_quad_names = get_skew_quad_nodes(ring, return_names=True)
skew_quad_strengths = np.zeros(len(skew_quad_nodes))
skew_quad_strengths[0] = 0.05

# Match without space charge
env = Envelope(eps, mode, ex_frac, mass, kin_energy, length=ring.getLength())
env.match_bare(ring, '2D')
set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths)

# Match with space charge
env.set_spacecharge(intensity)
solver_nodes = set_env_solver_nodes(ring, env.perveance, max_solver_spacing)
if match:
    print 'Matching.'
    env.match(ring, solver_nodes, tol=tol, verbose=verbose, method=method)
    
# Get linear transfer matrix
M = env.transfer_matrix(ring)

# Track envelope
env_monitor_node = AnalysisNode(0.0, 'env_monitor')
hf.add_node_at_start(ring, env_monitor_node)
env.track(ring, nturns_track, ntestparts, progbar=True)

# Save data
for key in ('env_params', 'testbunch_coords'):
    np.save(files[key], env_monitor_node.get_data(key, 'all_turns'))
np.save(files['transfer_matrix'], M)
np.savetxt('_output/data/mode.txt', [mode])