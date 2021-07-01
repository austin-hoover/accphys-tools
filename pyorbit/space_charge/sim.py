"""
This script tracks a coasting beam through a symmetric FODO lattice. The bunch
coordinates and transverse covariance matrix are saved after each cell.
"""

import sys
import numpy as np
from scipy import optimize as opt
from tqdm import trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf

# Local
from matching import Matcher

    
# Settings
#------------------------------------------------------------------------------
# Lattice
mu_x0 = 90.0 # horizontal tune [deg]
mu_y0 = 90.0 # vertical tune [deg]
cell_length = 5.0 # [m]
n_cells = 100

# Initial bunch
n_parts = 128000 # number of macro particles
mass = 0.93827231 # particle mass [GeV/c^2]
kin_energy = 1.0 # particle kinetic energy [GeV/c^2]
bunch_length = 150.0 # [m]
eps_x = 20e-6 # [m rad]
eps_y = 20e-6 # [m rad]
mu_x = 45.0 # depressed horizontal tune [deg]
mu_y = 45.0 # depressed vertical tune [deg]
bunch_kind = 'gaussian' 

# Space charge solver
max_solver_spacing = 0.1 # [m]
min_solver_spacing = 1e-6 # [m]
gridpts = (128, 128, 1) # (x, y, z)


# Generate rms matched distribution
# ------------------------------------------------------------------------------
lattice = hf.fodo_lattice(mu_x0, mu_y0, cell_length, fill_fac=0.5, start='quad')
matcher = Matcher(lattice, kin_energy, eps_x, eps_y)

print 'Setting depressed tunes.'
perveance = matcher.set_tunes(mu_x, mu_y, verbose=2)
intensity = hf.get_intensity(perveance, mass, kin_energy, bunch_length)
print 'Matched beam:'
print '    Perveance = {:.3e}'.format(perveance)
print '    Intensity = {:.3e}'.format(intensity)
print '    Zero-current tunes:', mu_x0, mu_y0
print '    Depressed tunes:', matcher.tunes()

print 'Generating bunch.'
kws = dict()
if bunch_kind == 'gaussian':
    kws['cut_off'] = 3.0
bunch, params_dict = hf.coasting_beam(bunch_kind, n_parts, matcher.twiss(), (eps_x, eps_y), 
                                      bunch_length, mass, kin_energy, intensity, **kws)

# Add space charge nodes
lattice.split(max_solver_spacing)    
calc2p5d = SpaceChargeCalc2p5D(*gridpts)
sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

# Add analysis nodes
monitor_node = AnalysisNode(0.0, kind='bunch_monitor')
stats_node = AnalysisNode(0.0, kind='bunch_stats')
hf.add_node_at_start(lattice, stats_node)
hf.add_node_at_start(lattice, monitor_node)

print 'Tracking bunch...'
hf.track_bunch(bunch, params_dict, lattice, n_cells)

# Save data
moments = stats_node.get_data('bunch_moments', 'all_turns')
np.save('data/moments.npy', moments)
coords = monitor_node.get_data('bunch_coords', 'all_turns')
np.save('data/coords.npy', coords)
