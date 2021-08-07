"""
This script examines the effect of skew quadrupoles in the SNS ring.
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
kin_energy = 1.0 # GeV/c^2
intensity = 1.5e14
bunch_length = 150.0 # m
nparts = int(1e5)
n_test_parts = 100
n_turns_track = 100
track_bunch = False
store_bunch_coords = False

# Lattice
latfile = '_latfiles/SNSring_noRF_nux6.18_nuy6.18.lat'
latseq = 'rnginj'
fringe = False

# Initial beam
mode = 2
eps = 50e-6 # intrinsic emitance
eps_x_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 1.0 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
match = True 
tol = 1e-4 
verbose = 2
perturb_radius = 0.0
method = 'lsq'
kws = dict(verbose=2, xtol=1e-15, gtol=1e-15, ftol=1e-15)

# Output data locations
files = {
    'env_params': '_output/data/envelope/env_params.npy',
    'testbunch_coords': '_output/data/envelope/testbunch_coords.npy',
    'bunch_coords': '_output/data/bunch/bunch_coords.npy',
    'bunch_moments': '_output/data/bunch/bunch_moments.npy',
    'transfer_matrix': '_output/data/transfer_matrix.npy'
}

delete_files_not_folders('_output/')


ring = hf.lattice_from_file(latfile, latseq, fringe)



# Configure skew quads
#------------------------------------------------------------------------------
skew_quads = dict()
for node in ring.getNodes():
    name = node.getName()
    if name.startswith('qsc'):
        node.setParam('skews', [0, 1])
        skew_quads[name] = node
    
def set_skew_quad_strength(node_name, strength):
    skew_quads[node_name].setParam('kls', [0.0, strength])
    
strength = 0.01
for letter in ['a', 'b', 'c', 'd']:
    for number in ['01', '09']:
        name = 'qsc_{}{}'.format(letter, number)
        set_skew_quad_strength(name, strength)
        
        
# Envelope
#------------------------------------------------------------------------------
# Create envelope matched to bare lattice
env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
env.match_bare(ring, '4D') # if '4D', and unequal tunes, beam will have 0 area

# Match 
env.set_intensity(intensity)
solver_nodes = set_env_solver_nodes(ring, env.perveance, max_solver_spacing)
if match and intensity > 0:
    print 'Matching.'
    env.match(ring, solver_nodes, tol=tol, method=method, **kws)
if perturb_radius > 0:
    print 'Perturbing envelope (radius = {:.2f}).'.format(perturb_radius)
    env.perturb(perturb_radius)
init_params = np.copy(env.params)

env.print_twiss4D()


# Get linear transfer matrix
M = env.transfer_matrix(ring)
mux, muy = 360 * env.tunes(ring)
print 'Transfer matrix is {}stable.'.format('' if twiss.is_stable(M) else 'un')
print '    mux, muy = {:.3f}, {:.3f} deg'.format(mux, muy)

# Track envelope
print 'Tracking envelope.'
env_monitor_node = AnalysisNode(0.0, 'env_monitor')
hf.add_node_at_start(ring, env_monitor_node)
env.track(ring, n_turns_track, n_test_parts, progbar=True)

# Save data
for key in ('env_params', 'testbunch_coords'):
    np.save(files[key], env_monitor_node.get_data(key, 'all_turns'))
np.save(files['transfer_matrix'], M)
np.savetxt('_output/data/mode.txt', [mode])