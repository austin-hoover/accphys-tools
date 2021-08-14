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
from orbit.utils.general import delete_files_not_folders

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_analysis import reconstruct
from phase_controller import PhaseController


# Settings
#------------------------------------------------------------------------------
# Beam
n_parts = 100000
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 1.5e14
beam_input_file = '_input/init_dist_128K.dat'

# Space charge solver
use_space_charge = False
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

# Lattice
madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 
              'alpha_y': 0.9749,
              'beta_x': 2.2991, 
              'beta_y': 14.2583}

# Scan parameters
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
steps_per_dim = 5 # number of steps for each dimension
method = 1
phase_coverage = 30.0 # [deg]
beta_lims = (40, 40) # (x, y)


# Initialize lattice
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)

# Add space charge nodes
if use_space_charge:
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# Add wire-scanner nodes
ws_nodes = dict()
for ws_name in ws_names:
    ws_node = WireScannerNode(name=ws_name)
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

Sigma0 = np.cov(X0.T)
stats0 = analysis.BunchStats(Sigma0)
print('Initial beam stats:')
print('Sigma =')
print(stats0.Sigma)
stats0.show(mm_mrad=True)

# Estimate the perveance (Maybe the rms length would be better? Or use
# the other definition of perveance in terms of the beam current?). This
# tells us whether space charge is relevant.
bunch_length = np.max(X0[:, 4]) - np.min(X0[:, 4])
line_density = intensity / bunch_length
print('perveance =', hf.get_perveance(mass, kin_energy, line_density))
print('4 * (eps_x / x_rms)**2 =', 4 * (stats0.eps_x**2 / stats0.Sigma[0, 0]))
print('4 * (eps_y / y_rms)**2 =', 4 * (stats0.eps_y**2 / stats0.Sigma[2, 2]))


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
        rec_node_name = None
        transfer_mats[ws_name].append(controller.transfer_matrix(rec_node_name, ws_name))
        moments[ws_name].append(ws_node.get_moments())
        phases[ws_name].append(controller.phase_adv(ws_name))
        
# Reconstruct covariance matrix.
transfer_mats_list, moments_list = [], []
for ws_name in ws_names:
    transfer_mats_list.extend(transfer_mats[ws_name])
    moments_list.extend(moments[ws_name])
    
Sigma = reconstruct(transfer_mats_list, moments_list)
stats = analysis.BunchStats(Sigma)
print('Reconstructed beam stats:')
print('Sigma =')
print(stats.Sigma)
stats.show(mm_mrad=True)

    
# Save data
#------------------------------------------------------------------------------
np.savetxt('_output/data/Sigma_rec.dat', Sigma)
np.savetxt('_output/data/Sigma0.dat', Sigma0)
np.savetxt('_output/data/X0.dat', X0)
for ws in ws_names:
    np.save('_output/data/{}/phases.npy'.format(ws), phases[ws])
    np.save('_output/data/{}/transfer_mats.npy'.format(ws), transfer_mats[ws])
    np.save('_output/data/{}/moments.npy'.format(ws), moments[ws])