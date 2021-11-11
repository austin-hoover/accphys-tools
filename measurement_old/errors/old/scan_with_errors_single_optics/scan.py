"""
This script simulates an emittance measurement in the RTBT using the Multiple 
Locations - Single Optics (MLSO) approach. In this approach, the machine optics 
are not changed at all. 

Error sources considered
------------------------
Energy
    Uncertainty in the beam energy is equivalent to a systematic quadrupole
    field error. This can apparently be measured in the control room to within
    a few MeV. A maximum error of 5 [MeV] is chosen.
Energy spread
    From Holmes 2018, the rms energy spread of the beam should be around 
    2 [MeV]. This was visually estimated from a figure in the paper.
Fringe fields
    These are turned on, but shouldn't have a large effect over this short
    distance.
Quad fields
    There may be some uncertainty in the linear quadrupole coefficients. We
    estimate about 1-2%.
Quad tilt angles
    The quadrupoles could be misaligned. We estimate 1 [mrad], but need to talk
    to someone from survey/alignment.
Space charge
    Assumed 1.5e14 intensity; 2D PIC solver.
Mismatched beam
    We're assuming that the beam has the same Twiss parameters as the lattice at
    the RTBT entrance. In reality, I think this will be a valid assumption. The
    reason is that the beam is painted, so it should automatically match to the
    lattice. The best way to test is to simulate painting, then transport the 
    final beam to the start of the RTBT. After doing this, we can compare the 
    beam Twiss parameters with the lattice Twiss parameters.
Wire-scanner resolution
    We include (1) random error on the tilt angle of the diagonal wire, and (2)
    random error in each bin count. A maximum 5% bin count error and maximum 
    1 [mrad] diagonal wire tilt error were selected, which seems reasonable.
    
    In reality, the best way to estimate the beam size error from the 
    wire-scanners may be to take a bunch of wire-scans at the same optics 
    settings, compute the beam size in each case, and report the standard 
    deviation. But this will include pretty much all sources of error, not just 
    the wire-scanner, so I'll have to think about it more.
"""
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pprint import pprint

from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode, WireScannerNode, add_analysis_node
from orbit.analysis.analysis import intrinsic_emittances, apparent_emittances
from orbit.envelope import DanilovEnvelope
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

sys.path.append('/Users/46h/Research/code/accphys/pyorbit/emittance_measurement')
from data_analysis import reconstruct
from utils import PhaseController


# Settings
#------------------------------------------------------------------------------
n_parts = 128000
n_trials = 100
n_meas = 12

# Lattice
latfile = '_input/rtbt_short.lat'
latseq = 'surv'
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24'
ws_bins = 25
diag_wire_angle = np.radians(45.0)

# Initial beam
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
rms_energy_spread = 0.002 # [GeV]
intensity = 1.5e14
bunch_length = 0.66 * 248.0 # [m]
beam_type = 'danilov'
eps = 40e-6 # nonzero intrinsice emittance eps_x + eps_y [m rad]
mode = 1
eps_x_frac = 0.5
eps_x, eps_y = eps_x_frac * eps, (1 - eps_x_frac) * eps
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (alpha_x)

# Space charge solver
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
solver_gridpoints = (128, 128, 1)

# Errors
errors = {
    'bin counts': True,
    'diag wire angle': True,
    'energy': True,
    'energy spread': True,
    'fringe fields': True,
    'twiss mismatch': True,
    'quad fields': True,
    'quad tilt angles': True,
    'space charge': True,
}
max_frac_alpha_error = 0.05
max_frac_beta_error = 0.05
max_frac_field_error = 0.02 
max_kin_energy_error = 0.005 # [GeV]
max_quad_tilt_angle = 1e-3 # rad
max_diag_wire_angle_error = np.radians(1.0) # [rad]
max_frac_bin_count_error = 0.05

print 'Errors:'
pprint(errors) 


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

def initialize_bunch(init_twiss):
    init_env = DanilovEnvelope(eps, mode, eps_x_frac, mass, kin_energy, 
                        length=bunch_length)
    alpha_x, alpha_y, beta_x, beta_y = init_twiss
    if errors['twiss mismatch']:
        alpha_x *= np.random.uniform(1 - max_frac_alpha_error, 1 + max_frac_alpha_error)
        alpha_y *= np.random.uniform(1 - max_frac_alpha_error, 1 + max_frac_alpha_error)
        beta_x *= np.random.uniform(1 - max_frac_beta_error, 1 + max_frac_beta_error)
        beta_y *= np.random.uniform(1 - max_frac_beta_error, 1 + max_frac_beta_error)
    init_env.set_twiss2D(alpha_x, alpha_y, beta_x, beta_y, eps_x_frac)
    init_env.set_twiss4D_param('nu', np.radians(100))
    X0 = init_env.generate_dist(n_parts)  
    deltaE = rms_energy_spread if errors['energy spread'] else 0.0
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    hf.dist_to_bunch(X0, bunch, bunch_length, deltaE)
    bunch.macroSize(int(intensity / n_parts))
    return bunch, params_dict


# Create lattice
#------------------------------------------------------------------------------
lattice = hf.lattice_from_file(latfile, latseq, fringe=errors['fringe fields'])
dummy_lattice = hf.lattice_from_file(latfile, latseq, fringe=False)
controller = PhaseController(dummy_lattice, init_twiss, mass, kin_energy, ref_ws_name)
default_quad_strengths = controller.get_quad_strengths()

# Add space charge nodes
if errors['space charge']:
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*solver_gridpoints)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# Add wire-scanner nodes
ws_nodes = []
for ws_name in ws_names:
    parent_node = lattice.getNodeForName(ws_name)
    ws_node = WireScannerNode(ws_bins, diag_wire_angle, name=ws_name)
    if errors['diag wire angle']:
        ws_node.set_diag_wire_angle_error(max_diag_wire_angle_error)
    if errors['bin counts']:
        ws_node.set_frac_bin_count_error(max_frac_bin_count_error)
    parent_node.addChildNode(ws_node, parent_node.ENTRANCE)
    ws_nodes.append(ws_node)
    
    
# Perform scan
#------------------------------------------------------------------------------
reconstructed_emittances = []

for trial in trange(n_trials):

    # Add quadrupole tilt errors.
    if errors['quad tilt angles']:
        for node in lattice.getNodes():
            if node.getType() == 'quad teapot':
                angle = np.random.uniform(-max_quad_tilt_angle, +max_quad_tilt_angle)
                node.setTiltAngle(angle)

    # Get scaling factor for quadrupole strengths to simulate devation from 
    # design energy.
    kin_energy_error = np.random.uniform(-max_kin_energy_error, 
                                         +max_kin_energy_error)
    ref_momentum = hf.get_pc(mass, kin_energy)
    true_momentum = hf.get_pc(mass, kin_energy + kin_energy_error)
    
    # Add error in quadrupole field strengths.
    quad_strengths = np.copy(default_quad_strengths)
    if errors['quad fields']:
        lb = (1 - max_frac_field_error) * quad_strengths
        ub = (1 + max_frac_field_error) * quad_strengths
        quad_strengths = np.random.uniform(lb, ub)
    if errors['energy']:
        quad_strengths *= (ref_momentum / true_momentum)
    controller.set_quad_strengths(quad_strengths)
    controller.apply_settings(lattice)

    # Track the beam `n_meas` times, each time recording the beam moments 
    # at the wire-scanners and the transfer matrices from the reconstruction
    # point to the wire-scanners.
    transfer_mats, moments = [], []
    for _ in trange(n_meas):
        bunch, params_dict = initialize_bunch(init_twiss)
        lattice.trackBunch(bunch, params_dict)
        for ws_name, ws_node in zip(ws_names, ws_nodes):
            moments.append(ws_node.get_moments())
            transfer_mats.append(controller.get_transfer_matrix(ws_name))

    # Reconstruct the beam moments and compute the apparent/intrinsic emittances.
    Sigma = reconstruct(transfer_mats, moments)
    Sigma *= 1e6 # [mm mrad]
    eps1, eps2 = intrinsic_emittances(Sigma)
    epsx, epsy = apparent_emittances(Sigma)
    reconstructed_emittances.append([eps1, eps2, epsx, epsy])
    
# Save the calculated emittances at every step
reconstructed_emittances = np.array(reconstructed_emittances)
print 'mean:', np.mean(reconstructed_emittances, axis=0)
print 'std:', np.std(reconstructed_emittances, axis=0)
np.save('reconstructed_emittances.npy', reconstructed_emittances)