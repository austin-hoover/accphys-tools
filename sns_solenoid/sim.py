import sys
import os

import numpy as np
from tqdm import trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import DanilovEnvelopeBunchMonitorNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import addTeapotDiagnosticsNodeAsChild
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.space_charge.envelope import DanilovEnvelope
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.envelope import set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf
from orbit.utils.general import ancestor_folder_path

sys.path.append(ancestor_folder_path(os.path.abspath(__file__), 'accphys'))
from tools.utils import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 0.0e14
n_turns = 500
n_parts = 10000

# Lattice
madx_file = '_input/SNSring_nux6.18_nuy6.18_solenoid.lat'
madx_file_nosolenoid = '_input/SNSring_nux6.24_nuy6.15.lat'
# madx_file_nosolenoid = '_input/SNSring_nux6.18_nuy6.18.lat'
madx_seq = 'rnginj'

switches = {
    'solenoid': False,
    'fringe': True,
}

# Initial beam
mode = 1
eps_l = 50e-6 # nonzero intrinsic emittance = eps_x + eps_y
eps_x_frac = 0.5 # eps_x / eps_l
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 1.0 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
match = False 
tol = 1e-7 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0.0 # If nonzero, perturb the matched envelope
method = 'auto' # 'lsq' or 'replace_by_avg'

delete_files_not_folders('_output/data/')
    
        
# Envelope
#------------------------------------------------------------------------------
# Create envelope matched to bare lattice without solenoid
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file_nosolenoid, madx_seq)
lattice.set_fringe(False)
env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, length=lattice.getLength())
env.match_bare(lattice, '2D') # if '4D', beam will be flat for uncoupled lattice


alpha_x, alpha_y, beta_x, beta_y = hf.twiss_at_entrance(lattice, mass, kin_energy)
    
# Match with space charge
env.set_intensity(intensity)
solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
if match:
    print 'Matching.'
    env.match(lattice, solver_nodes, tol=tol, verbose=verbose, method=method)
if perturb_radius > 0:
    print 'Perturbing envelope (radius = {:.2f}).'.format(perturb_radius)
    env.perturb(perturb_radius)
init_env_params = np.copy(env.params)
    
# Get linear transfer matrix
M = env.transfer_matrix(lattice)
nu1, nu2 = twiss.get_eigtunes(M)
print(nu1, nu2)
np.savetxt('_output/data/tunes.dat', [nu1, nu2])


# Bunch
#------------------------------------------------------------------------------
# Create lattice
lattice = TEAPOT_Lattice()
madx_file = madx_file if switches['solenoid'] else madx_file_nosolenoid
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(switches['fringe'])


M = hf.transfer_matrix(lattice, mass, kin_energy)
nu1, nu2 = twiss.get_eigtunes(M)
print(nu1, nu2)
np.savetxt('_output/data/eigtunes.dat', [nu1, nu2])



# Add space charge nodes
lattice.split(max_solver_spacing)    
if intensity > 0:
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

# Create bunch
env.params = init_env_params
bunch, params_dict = env.to_bunch(n_parts, no_env=True)

# Add analysis nodes
bunch_monitor_node = BunchMonitorNode(transverse_only=False)
bunch_stats_node = BunchStatsNode()
hf.add_node_at_start(lattice, bunch_stats_node)
hf.add_node_at_start(lattice, bunch_monitor_node)

tunes = TeapotTuneAnalysisNode("tune_analysis")
eta_x = eta_px = 0.
tunes.assignTwiss(beta_x, alpha_x, eta_x, eta_px, beta_y, alpha_y)
addTeapotDiagnosticsNodeAsChild(lattice, lattice.getNodes()[0], tunes)

# Track bunch
print 'Tracking bunch.'
for _ in trange(n_turns):
    lattice.trackBunch(bunch, params_dict)
    
# Save data
moments_tbt = [stats.moments for stats in bunch_stats_node.get_data()]
np.save('_output/data/moments.npy', moments_tbt)
np.save('_output/data/coords.npy', bunch_monitor_node.get_data())


bunch.dumpBunch('_output/data/bunch.dat')