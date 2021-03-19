"""
This script repeatedly tracks a beam through the RTBT, each time varying the 
optics and computing the phase advance, transfer matrix, and beam moments at 
the 5 wire-scanners. It does so for different beam intensities and beam
distributions (Gaussian, Waterbag, KV, Danilov).
"""

# Standard 
import sys
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
from utils import PhaseController


# Settings
#------------------------------------------------------------------------------
# General
intensities = np.linspace(0, 1.5e14, 5)
mass = 0.93827231 # GeV/c^2
kin_energy = 0.8 # GeV
bunch_length = 150.0 # [m]
nparts = int(1e5)
max_solver_spacing = 0.02
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
nsteps = 15 # number of steps for each dimension
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

# Initial envelope
ax0, ay0, bx0, by0 = init_twiss
env = Envelope(eps, mode=mode)
env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)

# Initial coordinate array
def get_coord_array(kind):
    if kind == 'danilov':
        ax0, ay0, bx0, by0 = init_twiss
        env = Envelope(eps, mode=mode)
        env.fit_twiss2D(ax0, ay0, bx0, by0, ex_frac)
        return env.generate_dist(nparts)
    else:
        constructors = {'kv':KVDist2D, 
                        'gaussian':GaussDist2D, 
                        'waterbag':WaterBagDist2D}
        (ax, ay, bx, by) = init_twiss
        twissX = TwissContainer(ax, bx, ex)
        twissY = TwissContainer(ay, by, ey)
        kws = {'cut_off':3} if kind == 'gaussian' else {} 
        dist_generator = constructors[kind](twissX, twissY, **kws)
        X = []
        for _ in range(nparts):
            X.append(dist_generator.getCoordinates())
        return np.array(X)

bunch_kinds = ['danilov', 'kv', 'gaussian']
X0 = {kind: get_coord_array(kind) for kind in bunch_kinds}

def reset_bunch(kind, intensity):
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    hf.dist_to_bunch(X0[kind], bunch, bunch_length)
    bunch.macroSize(intensity / nparts)
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

# Add bunch monitor nodes
bunch_monitor_nodes = {ws_name: add_analysis_node(lattice, ws_name, 'bunch_monitor')
                       for ws_name in ws_names}


# Perform scans
#------------------------------------------------------------------------------
phases = controller.get_phases_for_scan(phase_coverage, nsteps)

for scan_index, (nux, nuy) in enumerate(phases):
    print 'Scan {} of {}.'.format(scan_index, 2 * nsteps)
    print 'Setting phases: nux, nuy = {:.3f}, {:.3f}.'.format(nux, nuy)
    controller.set_ref_ws_phases(nux, nuy, max_betas, verbose=2)
    controller.apply_settings(lattice)
    controller.track_twiss()
    for kind in bunch_kinds:
        print '  Tracking {} bunch.'.format(kind)
        for intensity in intensities:
            print '    Intensity = {:.2e}'.format(intensity)
            bunch, params_dict = reset_bunch(kind, intensity)
            for node in bunch_monitor_nodes.values():
                node.clear_data()
            lattice.trackBunch(bunch, params_dict)
            for ws_name in ws_names:
                transfer_matrix = controller.get_transfer_matrix(ws_name)
                moments = ws_nodes[ws_name].get_moments()                
                handle = '{}_{}_I{:.3e}'.format(ws_name, kind, intensity)
                np.save('_output/data/{}_transfer_matrix.npy'.format(handle), transfer_matrix)
                np.save('_output/data/{}_moments.npy'.format(handle), transfer_matrix)
np.save('_output/data/phases.npy', phases)