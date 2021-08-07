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
from utils import delete_files_not_folders
from utils import PhaseController


# Settings
#------------------------------------------------------------------------------
# Beam
n_parts = 100000
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 0.0
bunch_length = 150.0 # [m]
beam_input_file = '_input/Bm_Parts_2M.txt'

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
steps_per_dim = 3 # number of steps for each dimension
method = 1
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

# Save default optics
twiss_df = pd.DataFrame(controller.tracked_twiss.copy(), 
                        columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('_output/data/twiss.dat', index=False)

# Save wire-scanner positions
ws_positions = [controller.node_position(ws_name) for ws_name in ws_names]
np.savetxt('_output/data/ws_positions.dat', ws_positions)


# Initialize beam
# ------------------------------------------------------------------------------
# Load from file (this was provide by Jeff Holmes)
print('Loading initial beam coordinates.')
X0 = np.loadtxt(beam_input_file) # [mm-mrad]

# Take random sample of particles.
n_samples = 100000
idx = np.random.choice(X0.shape[0], n_samples, replace=False)
X0 = X0[idx]

# # Artificial distribution
# X0 = np.random.normal(size=(20000, 6))
# X0[:, 0] *= 0.020
# X0[:, 1] *= 0.002
# X0[:, 2] *= 0.020
# X0[:, 3] *= 0.002
# X0[:, 4] = np.random.uniform(0, 250.0, size=(20000,))
# X0[:, 5] = 0.0

# Print initial beam statistics.
Sigma0 = np.cov(X0.T)
Sigma0 = Sigma0[:4, :4]
eps_1, eps_2 = analysis.apparent_emittances(Sigma0)
eps_x, eps_y = analysis.intrinsic_emittances(Sigma0)
alpha_x, alpha_y, beta_x, beta_y = analysis.twiss2D(Sigma0)
print('Initial beam:')
print('  eps_1, eps_2 = {} {} [mm mrad]'.format(eps_1, eps_2))
print('  eps_x, eps_y = {} {} [mm mrad]'.format(eps_x, eps_y))
print('  alpha_x, alpha_y = {} {} [mm mrad]'.format(alpha_x, alpha_y))
print('  beta_x, beta_y = {} {} [mm mrad]'.format(beta_x, beta_y))

# Convert to m-rad and fill bunch.
X0[:, :4] *= 1e-3
X0[:, 4] *= lattice.getLength() / (2 * np.pi)
bunch, params = hf.initialize_bunch(mass, kin_energy)
for i in range(n_parts):
    bunch.addParticle(0, 0, 0, 0, 0, 0)

def reset_bunch(bunch):
    """Restore bunch to its initial state."""
    for i, (x, xp, y, yp, z, dE) in enumerate(X0):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.z(i, z)
        bunch.dE(i, dE)

        
# Perform scan
#------------------------------------------------------------------------------
# Initialize dictionaries. Each dictionary holds a list of moments/transfer
# matrices at each wire-scanner.
transfer_mats, moments = dict(), dict()
ws_phases = dict()
for ws_name in ws_names:
    transfer_mats[ws_name] = []
    moments[ws_name] = []
    ws_phases[ws_name] = []

# Calculate the correct optics for each step in the scan. This allows us to 
# perform Monte Carlo simulations without recalculating at each step.
scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
quad_names = controller.ind_quad_names
quad_strengths_list = []
for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
    print('Calculating optics for scan {} of {}.'.format(scan_index, 2 * steps_per_dim))
    print('nux, nuy = {:.3f}, {:.3f}.'.format(nux, nuy))
    controller.set_phase_adv(ref_ws_name, nux, nuy, beta_lims, verbose=2)
    controller.track()
    quad_strengths = controller.quad_strengths(quad_names)
    quad_strengths_list.append(quad_strengths)
    

for scan_index, quad_strengths in enumerate(quad_strengths_list):
    
    # Set the correct lattice optics.
    controller.set_quad_strengths(quad_names, quad_strengths)

    # Track the bunch through the lattice.
    print('Tracking bunch.')
    reset_bunch(bunch)
    lattice.trackBunch(bunch, params)
    
    # Compute moments and transfer matrix at each wire-scanner.
    for ws_name in ws_names:
        ws_node = ws_nodes[ws_name]
        sig_xx, sig_yy, sig_xy = ws_node.get_moments()
        moments[ws_name].append([sig_xx, sig_yy, sig_xy])
        transfer_mats[ws_name].append(controller.transfer_matrix(ws_name))
        ws_phases[ws_name].append(controller.phase_adv(ws_name))

    # Save tracked Twiss parameters
    twiss_df = pd.DataFrame(controller.tracked_twiss, 
                            columns=['s','nux','nuy','ax','ay','bx','by'])
    twiss_df[['nux','nuy']] %= 1
    twiss_df.to_csv('_output/data/twiss_{}.dat'.format(scan_index), index=False)

            
# Save data
#------------------------------------------------------------------------------
for ws in ws_names:
    np.save('_output/data/{}/phases.npy'.format(ws), ws_phases[ws])
    np.save('_output/data/{}/transfer_mats.npy'.format(ws), transfer_mats[ws])
    np.save('_output/data/{}/moments.npy'.format(ws), moments[ws])
    
np.savetxt('_output/data/Sigma0.dat', Sigma0)
np.savetxt('_output/data/X0.dat', X0)
np.savetxt('_output/init_twiss.dat', 
           [init_twiss[key] for key in ['alpha_x', 'alpha_y', 'beta_x', 'beta_y']])