"""
Same as scan.py but tracks an envelope instead of a bunch. We therefore
can't include energy spread or fringe fields. We also have to estimate an
error on the beam moments <x^2>, <y^2>, and <xy> instead of including errors
in the wire-scanner angles and bin counts.
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
ex_frac = 0.5
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
    'beam moments': False,
    'energy': False,
    'twiss mismatch': False,
    'quad fields': False,
    'quad tilt angles': False,
    'space charge': False,
}
max_frac_twiss_error = 0.05
max_kin_energy_error = 0.003 # GeV
max_frac_field_error = 0.01
max_quad_tilt_angle = 1e-3 # rad
max_frac_beam_moments_error = 0.05
n_trials = 200


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

# Get correct magnet settings
dummy_lattice = hf.lattice_from_file(latfile, latseq)
controller = PhaseController(dummy_lattice, init_twiss, mass, kin_energy, ref_ws_name)

scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
quad_strengths_list = np.load('quad_strengths_list.npy')
# quad_strengths_list = []
# for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
#     fstr = 'Setting phases (scan {}/{}): nux, nuy = {:.3f}, {:.3f}.'
#     print fstr.format(scan_index, 2 * steps_per_dim, nux, nuy)
#     quad_strengths = controller.set_ref_ws_phases(nux, nuy, verbose=2)
#     quad_strengths_list.append(quad_strengths)


init_env = Envelope(eps, mode, ex_frac, mass, kin_energy, length=bunch_length)
init_env.set_intensity(intensity)
np.save('_output/data/env_params.npy', init_env.params)

def initialize_env(init_twiss):
    env = init_env.copy()
    if errors['twiss mismatch']:
        init_twiss = np.array(init_twiss)
        init_twiss *= np.random.uniform(1 - max_frac_twiss_error, 1 + max_frac_twiss_error, size=4)        
    ax0, ay0, bx0, by0 = init_twiss
    env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)
    env.set_twiss_param_4D('nu', np.radians(100))
    return env


# Create lattice
#------------------------------------------------------------------------------
lattice = hf.lattice_from_file(latfile, latseq, fringe=False)

# Add space charge nodes
if errors['space charge']:
    env_solver_nodes = set_env_solver_nodes(lattice, init_env.perveance, max_solver_spacing)
    
# Add monitor nodes
env_monitor_nodes = [add_analysis_node(lattice, ws_name, 'env_monitor')
                     for ws_name in ws_names]
    
    
# Perform scan
#------------------------------------------------------------------------------
keys = list(errors.keys())

for i in range(len(keys) + 1):
    
    if i < len(keys):
        errors[keys[i]] = True
    else:
        for key in keys:
            errors[key] = True   
    pprint(errors) 
        
    emittances_list, moments_list, transfer_mats_list = [], [], []

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

            # Track envelope
            for monitor_node in env_monitor_nodes:
                monitor_node.clear_data()
            env = initialize_env(init_twiss)
            env.track(lattice)

            # Compute moments and transfer matrix at each wire-scanner
            for ws_name, monitor_node in zip(ws_names, env_monitor_nodes):
                env.params = monitor_node.get_data('env_params')
                Sigma = env.cov()
                if errors['beam moments']:
                    Sigma *= np.random.uniform(1 - max_frac_beam_moments_error, 1 + max_frac_beam_moments_error)
                moments.append([Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]])
                transfer_mats.append(controller.get_transfer_matrix(ws_name))

        # Reconstruct beam moments
        Sigma = reconstruct(transfer_mats, moments)
        Sigma *= 1e6 # convert to mm mrad
        eps1, eps2 = intrinsic_emittances(Sigma)
        epsx, epsy = apparent_emittances(Sigma)
        emittances_list.append([eps1, eps2, epsx, epsy])
        moments_list.append(moments)
        transfer_mats_list.append(transfer_mats)

    emittances_list = np.array(emittances_list)
    print 'means:', np.mean(emittances_list, axis=0)
    print 'stds:', np.std(emittances_list, axis=0)
    np.save('_output/data/emittances_list_{}.npy'.format(i), emittances_list)
    np.save('_output/data/moments_list_{}.npy'.format(i), moments_list)
    np.save('_output/data/transfer_mats_list_{}.npy'.format(i), transfer_mats_list)
    
    for key in keys:
        errors[key] = False   
        
np.save('_output/data/keys.npy', keys)