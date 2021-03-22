"""
This script repeatedly tracks a beam through the RTBT, each time varying the 
optics and computing the phase advance, transfer matrix, and beam moments at 
the 5 wire-scanners. It does so for different beam intensities and beam
distributions (Gaussian, Waterbag, KV, Danilov).
"""

# Standard 
import sys
import time
# Third party
import numpy as np
from numpy import pi
import pandas as pd
from scipy import optimize as opt
from tqdm import tqdm
# PyORBIT
from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode, WireScannerNode, add_analysis_node, add_ws_node
from orbit.bunch_generators import TwissContainer, KVDist2D, GaussDist2D, WaterBagDist2D
from orbit.envelope import Envelope
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

sys.path.append('/Users/46h/Research/code/accphys/pyorbit_studies/measurement')
from utils import PhaseController, get_coord_array


# Settings
#------------------------------------------------------------------------------
# General
intensities = np.linspace(0, 5.0e14, 10)
mass = 0.93827231 # GeV/c^2
kin_energy = 0.811 # GeV
bunch_length = 150.0 # [m]
nparts = int(1e5)
max_solver_spacing = 0.2
min_solver_spacing = 0.00001
gridpts = (128, 128, 1)
latfile = '_input/rtbt.lat'
latseq = 'surv'

# Initial beam
beam_type = 'danilov'
eps = 20e-6 # nonzero intrinsice emittance ex + ey [m*rad]
mode = 1
ex_frac = 0.5 
ex, ey = ex_frac * eps, (1 - ex_frac) * eps
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)

# Scan
ws_names = ['ws02', 'ws20', 'ws21', 'ws23', 'ws24']
ref_ws_name = 'ws24' 
nsteps = 5 # number of steps for each dimension
wsbins = 50
phase_coverage = 180 # [deg]
max_betas = (40, 40) # (x, y)
diag_wire_angle = np.radians(45.0)


# Initialization
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

# Create phase controller
latt = hf.lattice_from_file(latfile, latseq)
controller = PhaseController(latt, init_twiss, mass, kin_energy, ref_ws_name)

# Save wire-scanner positions
ws_positions = [controller.get_node_position(ws) for ws in ws_names]
np.savetxt('_output/data/ws_positions.dat', ws_positions)

    
# Set up initial beam (to do: phase difference should be nonzero!)
#------------------------------------------------------------------------------
bunch_kinds = ['danilov', 'kv', 'gaussian']
X0 = {kind: get_coord_array(kind, init_twiss, (ex, ey), nparts) 
      for kind in bunch_kinds}
Sigma0 = {kind: np.cov(X0[kind].T) for kind in bunch_kinds}

def reset_bunch(kind, intensity):
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    hf.dist_to_bunch(X0[kind], bunch, bunch_length)
    bunch.macroSize(intensity / nparts if intensity > 0 else 1)
    return bunch, params_dict


# Create lattice
#------------------------------------------------------------------------------
lattice = hf.lattice_from_file(latfile, latseq)
lattice.split(max_solver_spacing)    
calc2p5d = SpaceChargeCalc2p5D(*gridpts)
setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    
# Add wire-scanner nodes
ws_nodes = {ws_name: add_ws_node(lattice, ws_name, wsbins, diag_wire_angle)
            for ws_name in ws_names}


# Perform scans
#------------------------------------------------------------------------------
phases = controller.get_phases_for_scan(phase_coverage, nsteps)
nscans = len(phases)
measurements_per_scan = len(ws_names)
total_measurements = nscans * measurements_per_scan

I_scan_M, I_scan_moments = {}, {}
for kind in bunch_kinds:
    I_scan_M[kind] = np.zeros((len(intensities), total_measurements, 4, 4))
    I_scan_moments[kind] = np.zeros((len(intensities), total_measurements, 3))
    
for scan_index, (nux, nuy) in enumerate(phases):
    print 'Scan {} of {}.'.format(scan_index + 1, nscans)
    print 'Setting phases: nux, nuy = {:.3f}, {:.3f}.'.format(nux, nuy)
    controller.set_ref_ws_phases(nux, nuy, max_betas, verbose=2)
    controller.track_twiss()
    controller.apply_settings(lattice)
    for kind in bunch_kinds:
        print '  Tracking {} bunch.'.format(kind)
        for i, intensity in enumerate(intensities):
            print '    Intensity = {:.2e}'.format(intensity)
            bunch, params_dict = reset_bunch(kind, intensity)
            lattice.trackBunch(bunch, params_dict)
            for ws_number, ws_name in enumerate(ws_names):    
                j = scan_index * measurements_per_scan + ws_number
                I_scan_M[kind][i, j] = controller.get_transfer_matrix(ws_name)
                I_scan_moments[kind][i, j] = ws_nodes[ws_name].get_moments()
                
for kind in bunch_kinds:
    np.save('_output/data/{}/I_scan_M.npy'.format(kind), I_scan_M[kind])
    np.save('_output/data/{}/I_scan_moments.npy'.format(kind), I_scan_moments[kind])
    np.save('_output/data/{}/X0.npy'.format(kind), X0[kind])   
    np.save('_output/data/{}/Sigma0.npy'.format(kind), np.cov(X0[kind].T))         
np.save('_output/data/phases.npy', phases)
np.save('_output/data/intensities.npy', intensities)