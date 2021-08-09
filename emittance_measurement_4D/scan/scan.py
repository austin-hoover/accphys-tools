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
from orbit.diagnostics import add_analysis_node
from orbit.diagnostics import add_analysis_nodes
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_analysis import reconstruct
from utils import delete_files_not_folders
from utils import PhaseController


def print_stats(Sigma):
    eps_1, eps_2 = analysis.apparent_emittances(Sigma)
    eps_x, eps_y = analysis.intrinsic_emittances(Sigma)
    eps_1, eps_2, eps_x, eps_y = 1e6 * np.array([eps_1, eps_2, eps_x, eps_y])
    alpha_x, alpha_y, beta_x, beta_y = analysis.twiss2D(Sigma)
    print('  eps_1, eps_2 = {} {} [mm mrad]'.format(eps_1, eps_2))
    print('  eps_x, eps_y = {} {} [mm mrad]'.format(eps_x, eps_y))
    print('  alpha_x, alpha_y = {} {} [rad]'.format(alpha_x, alpha_y))
    print('  beta_x, beta_y = {} {} [m/rad]'.format(beta_x, beta_y))
    

# Settings
#------------------------------------------------------------------------------
# Beam
n_parts = 100000
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 0.0e14
beam_input_file = '_input/init_dist_128K.dat'

# Space charge solver
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

# Lattice
madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 'alpha_y': 0.9749,
              'beta_x': 2.2991, 'beta_y': 14.2583}

# Scan parameters
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
steps_per_dim = 5 # number of steps for each dimension
method = 2
phase_coverage = 30.0 # [deg]
beta_lims = (40, 40) # (x, y)


# Initialize lattice
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)

# Add space charge nodes
if intensity > 0:
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# Add wire-scanner nodes
ws_nodes = dict()
for ws_name in ws_names:
    ws_node = WireScannerNode()
    parent_node = lattice.getNodeForName(ws_name)
    parent_node.addChildNode(ws_node, parent_node.ENTRANCE)
    ws_nodes[ws_name] = ws_node


# Initialize PhaseController
#------------------------------------------------------------------------------
controller = PhaseController(lattice, init_twiss, mass, kin_energy, ref_ws_name)

# Save default optics.
twiss_df = pd.DataFrame(controller.tracked_twiss.copy(), 
                        columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('_output/data/twiss.dat', index=False)

# Save wire-scanner positions.
ws_positions = [controller.node_position(ws_name) for ws_name in ws_names]
np.savetxt('_output/data/ws_positions.dat', ws_positions)


# Initialize beam
#------------------------------------------------------------------------------
print('Loading initial beam coordinates.')
X0 = np.loadtxt(beam_input_file)
Sigma0 = np.cov(X0.T)
Sigma0 = Sigma0[:4, :4]
print('Initial beam stats:')
print('Sigma =')
print(Sigma0)
print_stats(Sigma0)

# Initialize Bunch.
bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
for (x, xp, y, yp, z, dE) in X0:
    bunch.addParticle(x, xp, y, yp, z, dE)

def reset_bunch(bunch):
    for i, (x, xp, y, yp, z, dE) in enumerate(X0):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, dE)


# Perform scan
#------------------------------------------------------------------------------
transfer_mats = dict() # transfer matrix at each wire-scanner
moments = dict() # moments at each wire-scanner
phases = dict() # phase advance at each wire-scanner
for ws_name in ws_names:
    transfer_mats[ws_name] = []
    moments[ws_name] = []
    phases[ws_name] = []

scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
    print('Calculating optics for scan {} of {}.'.format(scan_index, 2 * steps_per_dim))
    print('nux, nuy = {:.3f}, {:.3f}.'.format(nux, nuy))
    controller.set_phase_adv(ref_ws_name, nux, nuy, beta_lims, verbose=1)
    controller.track()
    print('Tracking bunch.')
    reset_bunch(bunch)
    lattice.trackBunch(bunch, params_dict)
    print('Collecting measurements.')
    for ws_name in ws_names:
        ws_node = ws_nodes[ws_name]
        sig_xx, sig_yy, sig_xy = ws_node.get_moments()
        transfer_mats[ws_name].append(controller.transfer_matrix(ws_name))
        moments[ws_name].append(ws_node.get_moments())
        phases[ws_name].append(controller.phase_adv(ws_name))
        
# Reconstruct covariance matrix.
active_ws_names = ws_names[:]
max_n_meas = 20
transfer_mats_list, moments_list = [], []
for ws_name in active_ws_names:
    transfer_mats_list.extend(transfer_mats[ws_name][:max_n_meas])
    moments_list.extend(moments[ws_name][:max_n_meas])
    
Sigma_rec = reconstruct(transfer_mats_list, moments_list)
print('Reconstructed beam stats:')
print('Sigma =')
print(Sigma_rec)
print_stats(Sigma_rec)
    
# Save data
#------------------------------------------------------------------------------
np.savetxt('_output/data/Sigma_rec.dat', Sigma_rec)
np.savetxt('_output/data/Sigma0.dat', Sigma0)
np.savetxt('_output/data/X0.dat', X0)
for ws in ws_names:
    np.save('_output/data/{}/phases.npy'.format(ws), phases[ws])
    np.save('_output/data/{}/transfer_mats.npy'.format(ws), transfer_mats[ws])
    np.save('_output/data/{}/moments.npy'.format(ws), moments[ws])