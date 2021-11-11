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
from orbit.envelope import Envelope
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf

sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

sys.path.append('/Users/46h/Research/code/accphys/pyorbit/measurement')
from utils import PhaseController
from data_analysis import reconstruct


# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # GeV/c^2
kin_energy = 0.8 # GeV
energy_spread = 0.002 # GeV
intensity = 1.5e14
bunch_length = 160.0 # m
nparts = int(1e5)
max_solver_spacing = 1.0
min_solver_spacing = 0.00001
solver_gridpoints = (128, 128, 1)
latfile = '_input/rtbt.lat'
latseq = 'surv'

# Initial beam
beam_type = 'danilov'
eps = 40e-6 # nonzero intrinsice emittance ex + ey [m*rad]
mode = 1
ex_frac = 3.0 / 8.0
ex, ey = ex_frac * eps, (1 - ex_frac) * eps
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)

# Scan parameters
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
steps_per_dim = 6 # number of steps for each dimension
method = 2
ws_bins = 25
phase_coverage = 180 # [deg]
max_betas = (40, 40) # (x, y)
diag_wire_angle = np.radians(30.0)

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
max_frac_twiss_error = 0.10
max_kin_energy_error = 0.003 # GeV
max_frac_field_error = 0.02 
max_quad_tilt_angle = 2e-3 # rad
max_diag_wire_angle_error = np.radians(1.0) # rad
max_frac_bin_count_error = 0.05
n_trials = 50

print 'Errors:'
pprint(errors) 


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')


# Get correct magnet settings
dummy_lattice = hf.lattice_from_file(latfile, latseq)
controller = PhaseController(dummy_lattice, init_twiss, mass, kin_energy, ref_ws_name)

scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
# quad_strengths_list = []
# for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
#     fstr = 'Setting phases (scan {}/{}): nux, nuy = {:.3f}, {:.3f}.'
#     print fstr.format(scan_index, 2 * steps_per_dim, nux, nuy)
#     quad_strengths = controller.set_ref_ws_phases(nux, nuy, verbose=2)
#     quad_strengths_list.append(quad_strengths)
quad_strengths_list = np.load('quad_strengths_list.npy')


def initialize_bunch(init_twiss):
    init_env = Envelope(eps, mode, ex_frac, mass, kin_energy, length=0.66*248.0)
    if errors['twiss mismatch']:
        init_twiss = np.array(init_twiss)
        init_twiss *= np.random.uniform(1 - max_frac_twiss_error, 1 + max_frac_twiss_error, size=4)        
    ax0, ay0, bx0, by0 = init_twiss
    init_env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)
    init_env.set_twiss_param_4D('nu', np.radians(100))
    X0 = init_env.generate_dist(nparts)
    if errors['energy spread']:
        deltaE = energy_spread
    else:
        deltaE = 0.0    
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    hf.dist_to_bunch(X0, bunch, bunch_length, deltaE)
    bunch.macroSize(intensity / nparts)
    return bunch, params_dict


# Create lattice
#------------------------------------------------------------------------------
lattice = hf.lattice_from_file(latfile, latseq, fringe=errors['fringe fields'])

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

    # Tilt quads
    if errors['quad tilt angles']:
        for node in lattice.getNodes():
            if node.getType() == 'quad teapot':
                angle = np.random.uniform(-max_quad_tilt_angle, max_quad_tilt_angle)
                node.setTiltAngle(angle)

    # Get scaling factor for quad strengths to simulate devation from design energy
    kin_energy_error = np.random.uniform(-max_kin_energy_error, max_kin_energy_error)
    ref_momentum = hf.get_pc(mass, kin_energy)
    true_momentum = hf.get_pc(mass, kin_energy + kin_energy_error)


    # Collect data
    transfer_mats, moments = [], []
    for quad_strengths in quad_strengths_list:

        if errors['quad fields']:
            lb = (1 - max_frac_field_error) * quad_strengths
            ub = (1 + max_frac_field_error) * quad_strengths
            quad_strengths = np.random.uniform(lb, ub)

        if errors['energy']:
            quad_strengths *= (ref_momentum / true_momentum)

        controller.set_quad_strengths(quad_strengths)
        controller.apply_settings(lattice)

        # Track bunch
        bunch, params_dict = initialize_bunch(init_twiss)
        lattice.trackBunch(bunch, params_dict)

        # Compute moments and transfer matrix at each wire-scanner
        for ws_name, ws_node in zip(ws_names, ws_nodes):
            moments.append(ws_node.get_moments())
            transfer_mats.append(controller.get_transfer_matrix(ws_name))

    # Reconstruct beam moments
    Sigma = reconstruct(transfer_mats, moments)
    Sigma *= 1e6 # convert to mm mrad
    eps1, eps2 = intrinsic_emittances(Sigma)
    epsx, epsy = apparent_emittances(Sigma)
    reconstructed_emittances.append([eps1, eps2, epsx, epsy])
    
    
reconstructed_emittances = np.array(reconstructed_emittances)
print 'mean:', np.mean(reconstructed_emittances, axis=0)
print 'std:', np.std(reconstructed_emittances, axis=0)
np.save('reconstructed_emittances.npy', reconstructed_emittances)