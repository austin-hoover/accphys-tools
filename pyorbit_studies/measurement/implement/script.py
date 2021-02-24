"""
This script repeatedly tracks a beam through the RTBT, computing the beam 
moments, phase advance, and transfer matrix at the 5 wire-scanners.
"""

# Standard 
import sys
import fileinput
import subprocess
# Third party
import numpy as np
import pandas as pd
from scipy import optimize as opt
from tqdm import tqdm
# PyORBIT
from bunch import Bunch
from orbit.analysis import AnalysisNode
from orbit.envelope import Envelope
from orbit.matrix_lattice import BaseMATRIX
from orbit_utils import Matrix
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

from utils import (
    run_madx, 
    get_matlat,
    unpack,
    delete_first_line,
    MadxController, Scanner
)


# Settings
#------------------------------------------------------------------------------
# General
mass = 0.93827231 # GeV/c^2
kin_energy = 1.0 # GeV/c^2
intensity = 1e14
delete_files_not_folders('_output')

# Scan
script = 'RTBT.madx'
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ws_name = 'ws24' # phase advances are scanned at this WS
beta_max = (40, 40) # (x, y)
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)
# Deviations from default phase advance at chosen WS (normalized by 2pi)
delta_mu_list = np.linspace(-0.15, 0.15, 20)
    
    
# Setup
#------------------------------------------------------------------------------
# Set defaults in MADX script
madx_controller = MadxController(script, init_twiss)
madx_controller.set_ws(ws_name)
madx_controller.set_max_beta(*beta_max)
madx_controller.set_delta_mu(0, 0)

# Save default optics
lattice = madx_controller.create_lattice(hide_output=True)
scanner = Scanner(lattice, init_twiss, mass, kin_energy)
twiss = scanner.track_twiss()
twiss_df = pd.DataFrame(twiss, columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('_output/data/twiss.dat', index=False)

# Create one lattice for each phase advance at chosen WS
lattices, hide_output = [], True
print 'Creating lattices for horizontal scan.'
for delta_mu in tqdm(delta_mu_list):
    madx_controller.set_delta_mu(delta_mu, 0)
    lattice = madx_controller.create_lattice(hide_output)
    lattices.append(lattice)
print 'Creating lattices for vertical scan.'
for delta_mu in tqdm(delta_mu_list):
    madx_controller.set_delta_mu(0, delta_mu) 
    lattice = madx_controller.create_lattice(hide_output)
    lattices.append(lattice)
    
    
# Phase scan
#------------------------------------------------------------------------------
env = Envelope(eps=20e-6)
ax0, ay0, bx0, by0 = init_twiss
env.fit_twiss2D(ax0, ay0, bx0, by0, 0.5)
env.advance_phase(mux=0, muy=0)
env_params0 = np.copy(env.params)
Sigma0 = env.cov()


def init_dict():
    return {name:[] for name in ws_names}

transfer_mats = init_dict()
cov_mats = init_dict()
phase_advances = init_dict()
env_params = init_dict()

print 'Scanning.'
for lattice in tqdm(lattices):
    scanner = Scanner(lattice, init_twiss, mass, kin_energy)  
    env.params = env_params0
    analysis_nodes = {}
    for name in ws_names:
        phase_advances[name].append(scanner.get_ws_phase(name))
        transfer_mats[name].append(scanner.get_transfer_matrix(name))
        analysis_nodes[name] = scanner.add_analysis_node(name, 'env_monitor')
    env.track(lattice)
    for name in ws_names:
        env.params = analysis_nodes[name].get_data('env_params') 
        Sigma = env.cov()
        cov_mats[name].append(Sigma)  
        env_params[name].append(env.params)

np.save('_output/data/Sigma0.npy', Sigma0)        
for name in ws_names:
    np.save('_output/data/phase_advances_{}.npy'.format(name), phase_advances[name])
    np.save('_output/data/transfer_mats_{}.npy'.format(name), transfer_mats[name])
    np.save('_output/data/cov_mats_{}.npy'.format(name), cov_mats[name])
    np.save('_output/data/env_params_{}.npy'.format(name), env_params[name])

# # The following is to track a real bunch
# analysis_node = AnalysisNode(ws_position, 'bunch_stats', mm_mrad=False)
# ws_node.addChildNode(analysis_node, ws_node.ENTRANCE)
# bunch, params_dict = env.to_bunch(nparts=1e5, no_env=True)
# lattice.trackBunch(bunch, params_dict)
# Sigma = analysis_node.get_data('bunch_cov')  