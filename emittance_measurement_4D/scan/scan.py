from __future__ import print_function
import sys
import os

import numpy as np
import pandas as pd

from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import analysis
from orbit.analysis import AnalysisNode
from orbit.analysis import WireScannerNode
from orbit.analysis import add_analysis_node
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils.orbit_mpi_utils import bunch_orbit_to_pyorbit
from orbit.utils import helper_funcs as hf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import delete_files_not_folders
from utils import PhaseController


# Settings
#------------------------------------------------------------------------------
n_parts = 100000
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 0.0
bunch_length = 150.0 # [m]
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

lattice_file = '_input/rtbt.lat'
lattice_seq = 'whole1'
beam_input_file = '_input/Bm_Parts_2M.txt'
init_twiss = {'alpha_x': -0.25897, 'alpha_y': 0.9749,
              'beta_x': 2.2991, 'beta_y': 14.2583}

# Scan parameters
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
steps_per_dim = 6 # number of steps for each dimension
method = 2
phase_coverage = 30.0 # [deg]
beta_lims = (40, 40) # (x, y)


# Initialize lattice
#------------------------------------------------------------------------------
lattice = hf.lattice_from_file(lattice_file, lattice_seq)

# Add space charge nodes
if intensity > 0:
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# # Add wire-scanner nodes
# ws_nodes = {}
# for ws in ws_names:
#     parent_node = lattice.getNodeForName(ws)
#     ws_node = WireScannerNode(ws_bins, diag_wire_angle, name=ws)
#     parent_node.addChildNode(ws_node, parent_node.ENTRANCE)
#     ws_nodes[ws] = ws_node


# Initialize PhaseController
#------------------------------------------------------------------------------
controller = PhaseController(lattice, init_twiss, mass, kin_energy, ref_ws_name)

# Save default optics
twiss_df = pd.DataFrame(controller.tracked_twiss.copy(), 
                        columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('_output/data/twiss.dat', index=False)

# Save wire-scanner positions
ws_positions = [controller.node_position(ws) for ws in ws_names]
np.savetxt('_output/data/ws_positions.dat', ws_positions)


# Initialize beam
# ------------------------------------------------------------------------------
# Load from file (this was provide by Jeff Holmes)
X0 = np.loadtxt(beam_input_file) # in mm-mrad
Sigma = np.cov(X0.T)
Sigma = Sigma[:4, :4]
eps_1, eps_2 = analysis.apparent_emittances(Sigma)
eps_x, eps_y = analysis.intrinsic_emittances(Sigma)
alpha_x, alpha_y, beta_x, beta_y = analysis.twiss2D(Sigma)
print('Initial beam:')
print('  eps_1, eps_2 = {} {} [mm mrad]'.format(eps_1, eps_2))
print('  eps_x, eps_y = {} {} [mm mrad]'.format(eps_x, eps_y))
print('  alpha_x, alpha_y = {} {} [mm mrad]'.format(alpha_x, alpha_y))
print('  beta_x, beta_y = {} {} [mm mrad]'.format(beta_x, beta_y))

# Take random samples.
n_samples = 100000
idx = np.random.choice(X0.shape[0], n_samples, replace=False)
X0 = X0[idx]

# Convert to m-rad and fill bunch.
X0[:, :4] *= 1e-3
X0[:, 4] *= lattice.getLength() / (2 * np.pi)
bunch, params_dict = hf.initialize_bunch(mass, kin_energy)

def reset_bunch(bunch):
    bunch.deleteAllParticles()
    for i, (x, xp, y, yp, z, dE) in enumerate(X0):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.z(i, z)
        bunch.dE(i, dE)


# # # Perform scan
# # #------------------------------------------------------------------------------
# # def init_dict():
# #     return {ws:[] for ws in ws_names}

# # transfer_mats = init_dict()
# # moments = init_dict()
# # coords = init_dict()
# # ws_phases = init_dict()
# # env_params = init_dict()

# # scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)




# # for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
# #     # Set phases at reference wire-scanner
# #     print 'Scan {} of {}.'.format(scan_index, 2 * steps_per_dim)
# #     print '  Setting phases: nux, nuy = {:.3f}, {:.3f}.'.format(nux, nuy)
# #     controller.set_ref_ws_phases(nux, nuy, max_betas, verbose=2)
# #     controller.track_twiss()
# #     controller.apply_settings(lattice)

# #     # Track bunch
# #     print '  Tracking bunch.'
# #     bunch, params_dict = reset_bunch()
# #     lattice.trackBunch(bunch, params_dict)
    
# #     # Compute moments and transfer matrix at each wire-scanner
# #     for ws in ws_names:
# #         moments[ws].append(ws_nodes[ws].get_moments())
# #         transfer_mats[ws].append(controller.get_transfer_matrix(ws))
# #         ws_phases[ws].append(controller.get_phases(ws))

# #     # Track envelope for comparison
# #     controller.apply_settings(env_lattice)
# #     for node in env_monitor_nodes.values():
# #         node.clear_data()
# #     env.params = init_env.params
# #     env.track(env_lattice)
# #     for ws in ws_names:
# #         env_params[ws].append(env_monitor_nodes[ws].get_data('env_params'))
        
# #     # Save tracked Twiss parameters
# #     twiss_df = pd.DataFrame(np.copy(controller.tracked_twiss), 
# #                             columns=['s','nux','nuy','ax','ay','bx','by'])
# #     twiss_df[['nux','nuy']] %= 1
# #     twiss_df.to_csv('_output/data/twiss_{}.dat'.format(scan_index), index=False)

            
# # # Save data
# # #------------------------------------------------------------------------------
# # for ws in ws_names:
# #     np.save('_output/data/{}/phases.npy'.format(ws), ws_phases[ws])
# #     np.save('_output/data/{}/transfer_mats.npy'.format(ws), transfer_mats[ws])
# #     np.save('_output/data/{}/moments.npy'.format(ws), moments[ws])
# #     np.save('_output/data/{}/env_params.npy'.format(ws), env_params[ws])
    
# # np.save('_output/data/Sigma0_env.npy', env.cov())
# # np.savetxt('_output/data/Sigma0.dat', np.cov(X0.T))
# # np.savetxt('_output/data/X0.dat', X0)
# # np.savetxt('_output/init_twiss.dat', init_twiss)