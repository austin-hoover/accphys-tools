"""
This script integrates the KV envelope equations over 500 cells for a few 
different space charge strengths. The point is to approach the 90 degree 
envelope instability stopband.
"""
import sys
import numpy as np
from scipy import optimize as opt
from tqdm import tqdm
from tqdm import trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf

# Local
from matching import Matcher


# Lattice
mu_x0 = 100.0 # horizontal tune [deg]
mu_y0 = 100.0 # vertical tune [deg]
cell_length = 5.0 # [m]
n_cells = 500
depressed_tunes = np.linspace(90.0, 71.0, 6)

# Initial bunch
mass = 0.93827231 # particle mass [GeV/c^2]
kin_energy = 1.0 # particle kinetic energy [GeV/c^2]
bunch_length = 150.0 # [m]
eps_x = 20e-6 # [m rad]
eps_y = 20e-6 # [m rad]

# Space charge solver
max_solver_spacing = 0.1 # [m]
min_solver_spacing = 1e-6 # [m]

# Tracking
lattice = hf.fodo_lattice(mu_x0, mu_y0, cell_length, fill_fac=0.5, start='quad')
matcher = Matcher(lattice, kin_energy, eps_x, eps_y)
sizes_list = []
for depressed_tune in tqdm(depressed_tunes):
    perveance = matcher.set_tunes(depressed_tune, depressed_tune, verbose=2)
    sizes = matcher.track(perveance, n_cells)
    sizes_list.append(sizes)

np.save('_output/data/sizes_list.npy', sizes_list)
np.savetxt('_output/data/depressed_tunes.dat', depressed_tunes)
