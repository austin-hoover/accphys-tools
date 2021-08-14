from __future__ import print_function
import sys
import os

import numpy as np
import pandas as pd

from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.diagnostics import analysis
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import WireScannerNode
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
from orbit.utils.general import delete_files_not_folders

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_analysis import reconstruct
from phase_controller import PhaseController
    

# Settings
#------------------------------------------------------------------------------
# Beam
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 0.0e14
beam_input_file = '_input/init_dist_128K.dat'

# Lattice
madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 'alpha_y': 0.9749,
              'beta_x': 2.2991, 'beta_y': 14.2583}
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']


# Initialize lattice
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)

ws_nodes = dict()
for ws_name in ws_names:
    ws_node = WireScannerNode()
    parent_node = lattice.getNodeForName(ws_name)
    parent_node.addChildNode(ws_node, parent_node.ENTRANCE)
    ws_nodes[ws_name] = ws_node
    
controller = PhaseController(lattice, init_twiss, mass, kin_energy)


# Initialize beam
#------------------------------------------------------------------------------
print('Loading initial beam coordinates.')
X0 = np.loadtxt(beam_input_file)
bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
for (x, xp, y, yp, z, dE) in X0:
    bunch.addParticle(x, xp, y, yp, z, dE)
bunch.macroSize(int(intensity / bunch.getSize()))
    
def reset_bunch(bunch):
    for i, (x, xp, y, yp, z, dE) in enumerate(X0):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, dE)

stats0 = analysis.BunchStats(X0)
print('Initial beam stats:')
print('Initial beam stats:')
print('Sigma =')
print(stats0.Sigma)
stats0.show()


# Perform simulation
#------------------------------------------------------------------------------
reset_bunch(bunch)
lattice.trackBunch(bunch, params_dict)
# for ws_name in ws_names:
ws_name = ws_names[0]
ws_node = ws_nodes[ws_name]
sig_xx, sig_yy, sig_xy = ws_node.get_moments()
transfer_mats[ws_name].append(controller.transfer_matrix(ws_name))
moments[ws_name].append(ws_node.get_moments())
phases[ws_name].append(controller.phase_adv(ws_name))

# # Reconstruct covariance matrix.
# active_ws_names = ws_names[:]
# max_n_meas = 20
# transfer_mats_list, moments_list = [], []
# for ws_name in active_ws_names:
#     transfer_mats_list.extend(transfer_mats[ws_name][:max_n_meas])
#     moments_list.extend(moments[ws_name][:max_n_meas])
    
# Sigma_rec = reconstruct(transfer_mats_list, moments_list)
# print('Reconstructed beam stats:')
# print('Sigma =')
# print(Sigma_rec)
# print_stats(Sigma_rec)
    
# # Save data
# #------------------------------------------------------------------------------
# np.savetxt('_output/data/Sigma_rec.dat', Sigma_rec)
# np.savetxt('_output/data/Sigma0.dat', Sigma0)
# np.savetxt('_output/data/X0.dat', X0)
# for ws in ws_names:
#     np.save('_output/data/{}/phases.npy'.format(ws), phases[ws])
#     np.save('_output/data/{}/transfer_mats.npy'.format(ws), transfer_mats[ws])
#     np.save('_output/data/{}/moments.npy'.format(ws), moments[ws])