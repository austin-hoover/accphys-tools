import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
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
intensity = 1.5e14
bunch_length = 0.6 * 248 # [m]
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
ex_frac = 0.3 
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


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')


# Get correct magnet settings
dummy_lattice = hf.lattice_from_file(latfile, latseq)
controller = PhaseController(dummy_lattice, init_twiss, mass, kin_energy, ref_ws_name)

scan_phases = controller.get_phases_for_scan(phase_coverage, steps_per_dim, method)
quad_strengths_list = []
for scan_index, (nux, nuy) in enumerate(scan_phases, start=1):
    fstr = 'Setting phases (scan {}/{}): nux, nuy = {:.3f}, {:.3f}.'
    print fstr.format(scan_index, 2 * steps_per_dim, nux, nuy)
    quad_strengths = controller.set_ref_ws_phases(nux, nuy, verbose=2)
    quad_strengths_list.append(quad_strengths)
    
np.save('quad_strengths_list.npy', quad_strengths_list)