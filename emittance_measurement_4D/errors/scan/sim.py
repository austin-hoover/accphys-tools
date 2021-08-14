"""This script simulates the 4D emittance measurement with errors.

The measurement is repeated N times, and the measured covariance matrix
at each trial is saved.

To do:
    * Use uniformly sampled errors instead of from Gaussian? Could get 
      large values from the Gaussian.
    * Quad tilt error.
    * Wire-scanner errors (tilt, noise)
    * Run a bunch of times
"""


from __future__ import print_function
import sys
import os
import collections
from tqdm import tqdm
from tqdm import trange

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
from orbit.utils.general import ancestor_folder_path
from orbit.utils.general import delete_files_not_folders

sys.path.append(ancestor_folder_path(os.path.abspath(__file__), 'accphys'))
from emittance_measurement_4D.data_analysis import reconstruct
from emittance_measurement_4D.phase_controller import PhaseController
from tools.utils import rand_rows


# Settings
#------------------------------------------------------------------------------
# Beam
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
intensity = 1.5e14
beam_input_file = '_input/init_dist_128K.dat'

# Lattice
madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 
              'alpha_y': 0.9749,
              'beta_x': 2.2991, 
              'beta_y': 14.2583}

# Space charge solver
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

# Scan
rec_node_name = 'start'
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
steps_per_dim = 5 # number of steps for each dimension
method = 1
phase_coverage = 30.0 # [deg]
beta_lims = (40, 40) # (x, y)

# Error values
rms_frac_quad_strength_err = 0.02
rms_kin_energy_err = 0.005 # [GeV]
max_ws_angle_err = 1.0 # [rad]
rms_frac_ws_count_err = 0.05

# Error sources to include
errors = {
    'energy spread': True,
    'kinetic energy': True,
    'fringe fields': True,
    'quad strengths': True,
    'space charge': False,
    'wire-scanner noise': True,
    'wire-scanner angle': True,
}

n_trials = 20


# Initialization
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(errors['fringe fields'])

# Add space charge nodes
if errors['space charge']:
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# Add wire-scanner nodes
ws_nodes = dict()
for ws_name in ws_names:
    rms_frac_count_err = None
    if errors['wire-scanner noise']:
        rms_frac_count_err = rms_frac_ws_count_err
    ws_node = WireScannerNode(name=ws_name, rms_frac_count_err=rms_frac_count_err)
    parent_node = lattice.getNodeForName(ws_name)
    parent_node.addChildNode(ws_node, parent_node.ENTRANCE)
    ws_nodes[ws_name] = ws_node
        
# Initialize beam
print('Loading initial beam coordinates.')
X0 = np.loadtxt(beam_input_file)
max_n_parts = 10000
X0 = rand_rows(X0, max_n_parts)
if not errors['energy spread']:
    X0[:, 5] = 0.0
    
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)
for (x, xp, y, yp, z, dE) in X0:
    bunch.addParticle(x, xp, y, yp, z, dE)
bunch.macroSize(int(intensity / bunch.getSize()))
params_dict = {'bunch': bunch}

print('Tracking beam to reconstruction point: {}.'.format(rec_node_name))
sublattice = hf.get_sublattice(lattice, stop_node_name=rec_node_name)
sublattice.trackBunch(bunch, params_dict)
X0 = analysis.bunch_coord_array(bunch)

def reset(bunch):
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

# Start the lattice from the reconstruction node.
controller = PhaseController(lattice, init_twiss, mass, kin_energy, ref_ws_name)
alpha_x, alpha_y, beta_x, beta_y = controller.twiss(rec_node_name)
init_twiss['alpha_x'] = alpha_x
init_twiss['alpha_y'] = alpha_y
init_twiss['beta_x'] = beta_x
init_twiss['beta_y'] = beta_y
lattice = hf.get_sublattice(lattice, start_node_name=rec_node_name)
controller = PhaseController(lattice, init_twiss, mass, kin_energy, ref_ws_name)


# Simulation
#------------------------------------------------------------------------------
# Determine correct optics for each step in the scan. 
scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
ind_quad_names = controller.ind_quad_names
quad_strengths_list = []
print('Calculating optics settings.')
for scan_index, (nux, nuy) in enumerate(tqdm(scan_phases), start=1):
    controller.set_phase_adv(ref_ws_name, nux, nuy, beta_lims, verbose=1)
    quad_strengths_list.append(controller.quad_strengths(ind_quad_names))

    
# Perform the measurement `n_trials` times.
print('Repeating measurement.')
Sigmas = []
for _ in trange(n_trials):
    
    if errors['kinetic energy']:
        kin_energy_err = np.random.normal(scale=rms_kin_energy_err)
        bunch.getSyncParticle().kinEnergy(kin_energy + kin_energy_err)
        
    assumed_ws_angle = np.radians(45.0)
    if errors['wire-scanner angle']:
        assumed_ws_angle += np.random.uniform(-max_ws_angle_err, max_ws_angle_err)
    
    tmats, moments = [], []
    
    for quad_strengths in quad_strengths_list:
        
        if errors['quad strengths']:
            quad_strengths *= (1.0 + np.random.normal(scale=rms_frac_quad_strength_err, 
                                                      size=len(quad_strengths)))
        
        controller.set_quad_strengths(ind_quad_names, quad_strengths)
        reset(bunch)
        lattice.trackBunch(bunch, params_dict)
        for ws_name in ws_names:
            ws_node = ws_nodes[ws_name]
            moments.append(ws_node.get_moments())
            tmats.append(controller.transfer_matrix(rec_node_name, ws_name))
            
    Sigma = reconstruct(tmats, moments)
    print('eps_1, eps_2 =', analysis.intrinsic_emittances(Sigma))
    Sigmas.append(Sigma)
    
    
emittances = []
for Sigma in Sigmas:
    eps_1, eps_2 = ba.intrinsic_emittances(Sigma)
    eps_x, eps_y = ba.apparent_emittances(Sigma)
    emittances.append([eps_x, eps_y, eps_1, eps_2])
emittances = np.array(emittances)
    
np.save('_output/data/Sigmas.npy', Sigmas)