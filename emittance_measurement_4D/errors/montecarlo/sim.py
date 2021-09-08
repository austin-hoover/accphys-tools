"""This script simulates the 4D emittance measurement with errors.

The measurement is repeated N times, and the measured covariance matrix
at each trial is saved.

Here we use the fixed-optics method. 
"""
from __future__ import print_function
import sys
import os
from tqdm import tqdm
from tqdm import trange

import numpy as np

from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.diagnostics import analysis
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import WireScannerNode
from orbit.diagnostics import add_analysis_node
from orbit.diagnostics import add_analysis_nodes
from orbit.lattice import AccNode
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
from orbit.utils.general import ancestor_folder_path
from orbit.utils.general import delete_files_not_folders

sys.path.append(ancestor_folder_path(os.path.abspath(__file__), 'accphys'))
from emittance_measurement_4D.analysis import reconstruct
from emittance_measurement_4D.analysis import is_physical_cov
from emittance_measurement_4D.phase_controller import PhaseController
from tools.utils import rand_rows


# Settings
#------------------------------------------------------------------------------
# Beam
max_n_parts = 100000
mass = 0.93827231 # [GeV/c^2]
kin_energy = 0.8 # [GeV]
intensity = 1.5e14
beam_input_file = '_input/init_dist_prod_128K.dat'

# Lattice
madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 
              'alpha_y': 0.9749,
              'beta_x': 2.2991, 
              'beta_y': 14.2583}
ws_names = ['ws20', 'ws21', 'ws23', 'ws24']
rec_node_name = 'ws02'
ref_ws_name = 'ws24'

# Space charge solver
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

# Error values
rms_frac_quad_strength_err = 0.01
rms_kin_energy_err = 0.010 # [GeV]
max_ws_angle_err = np.radians(1.0) # [rad]
rms_frac_ws_count_err = 0.05

# Error sources to include
errors = {
    'energy spread': False,
    'kinetic energy': False,
    'fringe fields': False,
    'quad strengths': False,
    'space charge': False,
    'wire-scanner noise': False,
    'wire-scanner angle': False,
}

n_trials = 25
modify_optics = False


# Initialization
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(errors['fringe fields'])

# Add space charge nodes
if errors['space charge'] and intensity > 0:
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
print('Keeping {} random samples.'.format(max_n_parts))
X0 = rand_rows(X0, max_n_parts)
if not errors['energy spread']:
    X0[:, 5] = 0.0

bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)
for (x, xp, y, yp, z, dE) in X0:
    bunch.addParticle(x, xp, y, yp, z, dE)
bunch.macroSize(int(intensity / bunch.getSize()) if intensity > 0 else 1)
params_dict = {'bunch': bunch}

print('Tracking beam to reconstruction point: {}.'.format(rec_node_name))
sublattice = hf.get_sublattice(lattice, stop_node_name=rec_node_name)
sublattice.trackBunch(bunch, params_dict)
X0 = analysis.bunch_coord_array(bunch)
Sigma0 = np.cov(X0.T)
stats0 = analysis.BunchStats(Sigma0)
print('Initial beam stats:')
print('Sigma =')
print(stats0.Sigma)
stats0.show(mm_mrad=True)

def reset(bunch):
    for i, (x, xp, y, yp, z, dE) in enumerate(X0):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, dE)

# Start the lattice from the reconstruction point.
controller = PhaseController(lattice, init_twiss, mass, kin_energy, ref_ws_name)

if modify_optics:
    print('Modifying phase advance at ws24')
    nux0, nuy0 = controller.phase_adv('ws24')
    controller.set_phase_adv('ws24', nux0 + (45./360), nuy0 - (25./360), 
                             beta_lims=(35.0, 35.0), verbose=0)

alpha_x, alpha_y, beta_x, beta_y = controller.twiss(rec_node_name)
init_twiss['alpha_x'] = alpha_x
init_twiss['alpha_y'] = alpha_y
init_twiss['beta_x'] = beta_x
init_twiss['beta_y'] = beta_y
lattice = hf.get_sublattice(lattice, start_node_name=rec_node_name)

controller = PhaseController(lattice, init_twiss, mass, kin_energy, ref_ws_name)
default_quad_strengths = controller.quad_strengths(controller.ind_quad_names)
tmats = [controller.transfer_matrix(None, ws_name) for ws_name in ws_names]


# Simulation
#------------------------------------------------------------------------------
Sigmas = []
emittances = []
n_fail = 0

print('Repeating measurement {} times.'.format(n_trials))
print('trial | eps_x  | eps_y  | eps_1  | eps_2')
print('-----------------------------------------')

for i in range(n_trials):
    
    quad_strengths = np.copy(default_quad_strengths)

    if errors['kinetic energy']:
        # Changing the bunch energy doesn't do anything since the quadrupole 
        # coefficients (B' / Brho) are what is set in PyORBIT instead of the
        # actual field. So it sort of automatically scales the quads depending
        # on the energy. Thus, we just scale the quad coefficents and keep the
        # beam energy fixed. 
        kin_energy_err = np.random.normal(scale=rms_kin_energy_err)        
        Brho_correct = hf.get_Brho(mass, kin_energy)
        Brho_err = hf.get_Brho(mass, kin_energy + kin_energy_err)
        quad_strengths *= (Brho_correct / Brho_err)
        controller.set_quad_strengths(controller.ind_quad_names, quad_strengths)

    assumed_ws_angle = np.radians(45.0)
    if errors['wire-scanner angle']:
        assumed_ws_angle += np.random.uniform(-max_ws_angle_err, max_ws_angle_err)

    quad_strengths = np.copy(default_quad_strengths)
    if errors['quad strengths']:
        quad_strengths *= (1.0 + np.random.normal(scale=rms_frac_quad_strength_err,
                                                  size=len(quad_strengths)))
        controller.set_quad_strengths(controller.ind_quad_names, quad_strengths)

    reset(bunch)
    lattice.trackBunch(bunch, params_dict)
    moments = [ws_nodes[ws_name].get_moments() for ws_name in ws_names]
    Sigma = reconstruct(tmats, moments)
    Sigma *= 1e6 # convert to [mm-mrad]

    if not is_physical_cov(Sigma):
        n_fail += 1
        print('Covariance matrix is unphysical.')
        continue

    eps_x, eps_y, eps_1, eps_2  = analysis.emittances(Sigma)
    emittances.append([eps_x, eps_y, eps_1, eps_2])
    Sigmas.append(Sigma)
    print('{:<5} | {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(i + 1, eps_x, eps_y, eps_1, eps_2))

fail_rate = float(n_fail) / n_trials

print('means:', np.mean(emittances, axis=0))
print('stds:', np.std(emittances, axis=0))
print('fail rate = {}'.format(fail_rate))

np.save('_output/data/Sigmas.npy', Sigmas)
np.save('_output/data/emittances.npy', emittances)
np.save('_output/data/Sigma0.npy', Sigma0)
np.savetxt('_output/data/fail_rate.dat', [fail_rate])