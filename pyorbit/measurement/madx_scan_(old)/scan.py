"""
This script repeatedly tracks a beam through the RTBT, each time varying the 
optics and computing the phase advance, transfer matrix, and beam moments at 
the 5 wire-scanners.
"""

# Standard 
import sys
import fileinput
import subprocess
import copy
# Third party
import numpy as np
import pandas as pd
from scipy import optimize as opt
from tqdm import tqdm
# PyORBIT
from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode, WireScannerNode
from orbit.envelope import Envelope
from orbit.matrix_lattice import BaseMATRIX
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

from utils import MadxController, PhaseScanner, add_ws_node, add_analysis_node


# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # GeV/c^2
kin_energy = 1.0 # GeV/c^2
intensity = 0.0e14
bunch_length = 150.0 # [m]
nparts = int(1e4)
max_solver_spacing = 0.02
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)

# Initial beam
beam_type = 'danilov'
eps = 20e-6 # nonzero intrinsice emittance ex + ey [m*rad]
mode = 1
ex_frac = 0.5 
ex, ey = ex_frac * eps, (1 - ex_frac) * eps
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)

# Scan
script = 'RTBT.madx'
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
beta_max = (40, 40) # (x, y)
nbins = 50
diag_wire_angle = np.radians(45.0)
# Deviations from default phase advance at chosen WS (normalized by 2pi)
ref_ws_name = 'ws24' 
delta_mu_list = np.linspace(-0.15, 0.15, 15)
    
# Create lattice for each step in scan
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

# Set defaults in MADX script
madx_controller = MadxController(script, init_twiss)
madx_controller.set_ws(ref_ws_name)
madx_controller.set_max_beta(*beta_max)
madx_controller.set_delta_mu(0, 0)

# Save default optics
lattice = madx_controller.create_lattice(hide_output=True)
scanner = PhaseScanner(lattice, init_twiss, mass, kin_energy)
twiss = scanner.track_twiss()
twiss_df = pd.DataFrame(twiss, columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('_output/data/twiss.dat', index=False)
ws_positions = []
for ws_name in ws_names:
    ws_node = lattice.getNodeForName(ws_name)
    ws_positions.append(lattice.getNodePositionsDict()[ws_node][0])
np.savetxt('_output/data/ws_positions.dat', ws_positions)

# Create lattice for each pair of phase advances
## `env_lattice` is for envelope tracking
lattices, env_lattices, hide_output = [], [], True
for scan in ('horizontal', 'vertical'):
    print 'Creating lattices for {} scan.'.format(scan)
    for delta_mu in tqdm(delta_mu_list):
        dmux, dmuy = (delta_mu, 0) if scan == 'horizontal' else (0, delta_mu)
        madx_controller.set_delta_mu(dmux, dmuy)
        madx_controller.generate_latfile(hide_output)
        lattices.append(madx_controller.read_latfile())
        env_lattices.append(madx_controller.read_latfile())
    
    
# Set up initial beam
#------------------------------------------------------------------------------

# Scale intensity to get perveance for 0.8 GeV
if intensity > 0.0:
    Q0 = hf.get_perveance(mass, 1.0, intensity/bunch_length)
    Q1 = hf.get_perveance(mass, 0.8, intensity/bunch_length)
    intensity *= (Q1 / Q0)
    print 'Simulating 0.8 GeV without changing optics.'
    print 'Scaled intensity to I = {:.02e}.'.format(intensity)

ax0, ay0, bx0, by0 = init_twiss
env = Envelope(eps, mode, ex_frac, mass, kin_energy, bunch_length, intensity)
env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)
env_params0 = np.copy(env.params)
np.save('_output/data/Sigma0_env.npy', env.cov())

if beam_type == 'danilov':
    X0 = env.generate_dist(nparts)
elif beam_type == 'gaussian':
    cut_off = 3
    bunch, params_dict = hf.coasting_beam(
        beam_type, nparts, init_twiss, (ex, ey), bunch_length, mass, 
        kin_energy, intensity, cut_off=cut_off)
    X0 = hf.dist_from_bunch(bunch)
    
np.savetxt('_output/data/X0.dat', X0)
np.savetxt('_output/data/Sigma0.dat', np.cov(X0.T))


# Perform scan
#------------------------------------------------------------------------------
def init_dict():
    return {ws_name:[] for ws_name in ws_names}

transfer_mats = init_dict()
moments = init_dict()
phase_advances = init_dict()
env_params = init_dict()

def reset_bunch():
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    hf.dist_to_bunch(X0, bunch, bunch_length)
    bunch.macroSize(intensity / nparts)
    return bunch, params_dict

print 'Scanning.'
for lattice, env_lattice in tqdm(zip(lattices, env_lattices)):
        
    # Get ideal transfer matrices and phase advances
    scanner = PhaseScanner(lattice, init_twiss, mass, kin_energy)  
    for ws_name in ws_names:
        phase_advances[ws_name].append(scanner.get_ws_phase(ws_name))
        transfer_mats[ws_name].append(scanner.get_transfer_matrix(ws_name))
        
    # Add space charge nodes
    bunch, params_dict = reset_bunch()
    lattice.split(max_solver_spacing)    
    if intensity > 0:
        calc2p5d = SpaceChargeCalc2p5D(*gridpts)
        setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

    # Add wire-scanner nodes
    ws_nodes = {ws_name: add_ws_node(lattice, nbins, diag_wire_angle, ws_name)
                for ws_name in ws_names}

    # Track bunch
    lattice.trackBunch(bunch, params_dict)
    for ws_name in ws_names:
        moments[ws_name].append(ws_nodes[ws_name].get_moments())
        
    # Track envelope
    set_env_solver_nodes(env_lattice, env.perveance, max_solver_spacing)
    analysis_nodes = {ws_name: add_analysis_node(env_lattice, ws_name, 'env_monitor') 
                      for ws_name in ws_names}
    env.params = env_params0
    env.track(env_lattice)
    for ws_name, analysis_node in analysis_nodes.items():
        env_params[ws_name].append(analysis_node.get_data('env_params'))
        
        
for ws_name in ws_names:
    np.save('_output/data/phase_advances_{}.npy'.format(ws_name), phase_advances[ws_name])
    np.save('_output/data/transfer_mats_{}.npy'.format(ws_name), transfer_mats[ws_name])
    np.save('_output/data/moments_{}.npy'.format(ws_name), moments[ws_name])
    np.save('_output/data/env_params_{}.npy'.format(ws_name), env_params[ws_name])