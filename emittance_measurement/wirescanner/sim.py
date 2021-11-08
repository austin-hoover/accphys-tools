"""RTBT wire-scanner emittance measurement simulation.

This script tracks a bunch from the RTBT entrance to the target. Wire-scanner
measurements are then simulated and used to reconstruct the transverse 
covariance matrix.

The initial bunch is loaded from a file in '_input/'. Alternatively, a
coasting Gaussian, KV, or Waterbag distribution can be generated from the 
design Twiss parameters or from user-supplied Twiss parameters.

The number of wire-scanner measurements, as well as the phase advances at each
measurement, are free parameters. There is also the option to include -- or 
not include -- the following effects and run a Monte Carlo simulation of the
measurement:
    * wire-scanner noise (very small in reality)
    * wire-scanner tilt angle error
    * energy error
    * quadrupole field errors
    * space charge
    * fringe fields
    
Since the initial bunch energy, mass, and intensity need to be hard-coded,
it might be easier to put this at the end of the painting script. Or we 
could make it read a file that is output from the painting script. 
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
from orbit.lattice import AccNode
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
from orbit.utils.general import ancestor_folder_path
from orbit.utils.general import delete_files_not_folders
from orbit.utils.general import load_stacked_arrays

# Local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from emittance_measurement.phase_controller import PhaseController
from emittance_measurement.analysis import reconstruct


# Settings
#------------------------------------------------------------------------------
n_trials = 1
beam_input_file = '_input/coords_rtbt.npz'
kin_energy = 0.8 # [GeV]
mass = 0.93827231 # [GeV/c^2]
final_intensity = 5.4e13

turns = list(range(0, 400, 50))
final_turn = turns[-1]

# Note: 1.5e11 is added to the intensities because turn 0 is a single
# minpulse.
intensities = final_intensity * np.divide(turns, float(final_turn)) + 1.5e11
print('Turns =', turns)
print('Intensities =', intensities)

madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 
              'alpha_y': 0.9749,
              'beta_x': 2.2991, 
              'beta_y': 14.2583}
ws_names = ['ws20', 'ws21', 'ws23', 'ws24']

# Space charge solver
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

# Errors
rms_frac_quad_strength_err = 0.01
rms_kin_energy_err = 0.005 # [GeV]
max_ws_angle_err = np.radians(1.0) # [rad]
rms_frac_ws_count_err = 0.05
errors = {
    'energy spread': False,
    'kinetic energy': False,
    'fringe fields': False,
    'quad strengths': False,
    'space charge': True,
    'wire-scanner noise': False,
    'wire-scanner angle': False,
}

# Initialization
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)

# Optimize magnet strengths for fixed-optics emittance reconstruction.
print('Modifying optics at WS24')
controller = PhaseController(lattice, init_twiss, mass, kin_energy, 'ws24')
nux0, nuy0 = controller.phase_adv('ws24')
nux = nux0 + (45. / 360.)
nuy = nuy0 - (45. / 360.)
controller.set_phase_adv('ws24', nux, nuy,
                         beta_lims=(35.0, 35.0), 
                         verbose=2)

# Save the default quadrupole strengths for later use.
default_quad_strengths = controller.quad_strengths(controller.ind_quad_names)

# Compute the transfer matrices for each node in the lattice.
tmats_dict = dict()
rec_node_names = []
for node in lattice.getNodes():
    node_name = node.getName()
    if 'Drift' in node_name:
        continue
    tmats_dict[node_name] = [controller.transfer_matrix(node_name, ws_name) 
                             for ws_name in ws_names]
    rec_node_names.append(node_name)
    
# Save node names.
file = open('_output/data/rec_node_names.txt', 'w')
for node_name in rec_node_names:
    file.write(node_name + '\n')
file.close()

# Save node positions.
rec_node_positions = []
node_pos_dict = lattice.getNodePositionsDict()
for node_name in rec_node_names:
    node = lattice.getNodeForName(node_name)
    pos_start, pos_stop = node_pos_dict[node]
    rec_node_positions.append([pos_start, pos_stop])
np.savetxt('_output/data/rec_node_positions.dat', rec_node_positions)

# Add space charge nodes.
lattice.split(max_solver_spacing)    
calc2p5d = SpaceChargeCalc2p5D(*gridpts)
sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
if (not errors['space charge']) or final_intensity == 0.:
    for node in sc_nodes:
        node.setCalculationOn(False)

# Add wire-scanner nodes.
ws_nodes = dict()
for ws_name in ws_names:
    rms_frac_count_err = None
    if errors['wire-scanner noise']:
        rms_frac_count_err = rms_frac_ws_count_err
    ws_node = WireScannerNode(name=ws_name, rms_frac_count_err=rms_frac_count_err)
    parent_node = lattice.getNodeForName(ws_name)
    parent_node.addChildNode(ws_node, parent_node.ENTRANCE)
    ws_nodes[ws_name] = ws_node
    
# Add bunch monitor nodes.
bunch_stats_nodes = dict()
for node_name in rec_node_names:
    bunch_stats_node = BunchStatsNode(mm_mrad=True)
    node = controller.lattice.getNodeForName(node_name)
    node.addChildNode(bunch_stats_node, node.ENTRANCE)
    bunch_stats_nodes[node_name] = bunch_stats_node
    
# Toggle fringe fields.
lattice.set_fringe(errors['fringe fields'])
    
# Load the bunch coordinates.
print('Loading initial beam coordinates.')
coords = None
if '.dat' in beam_input_file or '.txt' in beam_input_file:
    X0 = np.loadtxt(beam_input_file)
elif '.npy' in beam_input_file:
    X0 = np.load(beam_input_file)
elif '.npz' in beam_input_file:
    coords = load_stacked_arrays(beam_input_file)
    for t in range(len(coords)):
        coords[t][:, :4] *= 0.001
        
if coords is None:
    coords = [X0]
    
for frame in range(len(coords)):
    if not errors['energy spread']:
        coords[frame][:, 5] = 0.0
        
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)
params_dict = {'bunch': bunch} 

def set_bunch_coords(bunch, X):
    if bunch.getSize() == X.shape[0]:
        for i, (x, xp, y, yp, z, dE) in enumerate(X):
            bunch.x(i, x)
            bunch.xp(i, xp)
            bunch.y(i, y)
            bunch.yp(i, yp)
            bunch.z(i, z)
            bunch.dE(i, dE)
    else:
        bunch.deleteAllParticles()
        for (x, xp, y, yp, z, dE) in X:
            bunch.addParticle(x, xp, y, yp, z, dE)

        
# Simulate the measurement for each bunch in `coords`.
#------------------------------------------------------------------------------
for frame, X in enumerate(coords):
    
    # Initialize the bunch.
    set_bunch_coords(bunch, X)
    intensity = intensities[frame]
    turn = turns[frame]
    macro_size = max(1, int(intensity / bunch.getSize()))
    bunch.macroSize(macro_size)
    print('Bunch {}/{}'.format(frame + 1, len(coords)))
    print('Intensity = {:.3e}'.format(intensity))
    print('Turn = {}'.format(turn))
                    
    # For the reconstructed covariance matrices. 
    Sigmas_rec_dict = dict()
    for node_name in rec_node_names:
        Sigmas_rec_dict[node_name] = []
        
    # For the true (simulated) covariance matrices. 
    Sigmas_sim_dict = dict()
    for node_name in rec_node_names:
        Sigmas_sim_dict[node_name] = []

        
    print('Simulating measurement.')
    
    for trial_number in trange(n_trials):
        
        # Reset the lattice to its initial state.
        quad_strengths = np.copy(default_quad_strengths)
        for node_name in rec_node_names:
            bunch_stats_nodes[node_name].clear_data()

        # Add errors.
        if errors['kinetic energy']:
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

        # Track the bunch through the lattice.
        set_bunch_coords(bunch, X)
        controller.lattice.trackBunch(bunch, params_dict)

        # Estimate the moments from the wire-scanners.
        moments = []
        for ws_name in ws_names:
            ws_node = ws_nodes[ws_name]
            moments.append(ws_node.get_moments(assumed_ws_angle))
        moments = np.array(moments)
        moments *= 1e6

        # Reconstruct the covariance matrix at every node. 
        for node_name in rec_node_names:
            tmats = tmats_dict[node_name]
            Sigma, C = reconstruct(tmats, moments)        
            Sigmas_rec_dict[node_name].append(Sigma)
            
        # Store the actual covariance matrix at every node.
        for node_name in rec_node_names:
            bunch_stats_node = bunch_stats_nodes[node_name]
            bunch_stats = bunch_stats_node.get_data(0)
            Sigma = bunch_stats.Sigma
            Sigmas_sim_dict[node_name].append(Sigma)
            

    # Save the reconstructed covariance matrices.
    np.save('_output/data/Sigmas_rec_{}.npy'.format(frame), 
            [Sigmas_rec_dict[node_name] for node_name in rec_node_names])

    # Save the actual covariance matrices.
    np.save('_output/data/Sigmas_sim_{}.npy'.format(frame), 
            [Sigmas_sim_dict[node_name] for node_name in rec_node_names])
    
np.savetxt('_output/data/turns.dat', turns)
np.savetxt('_output/data/intensities.dat', intensities)