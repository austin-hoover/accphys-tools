import sys
import numpy as np
from scipy import optimize as opt
from tqdm import trange, tqdm

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode
from orbit.analysis.AnalysisNode import get_coords
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
n_cells = 500
cell_length = 5.0 # [m]
tunes_x = np.linspace(100.0, 90.0, n_cells) # [deg]
tunes_y = np.linspace(100.0, 90.0, n_cells) # [deg]

# Initial bunch
n_parts = 128000 # number of macro particles
mass = 0.93827231 # particle mass [GeV/c^2]
kin_energy = 1.0 # particle kinetic energy [GeV/c^2]
bunch_length = 150.0 # [m]
eps_x = 20e-6 # [m rad]
eps_y = 20e-6 # [m rad]
mu_x = 92.2 # initial depressed horizontal tune [deg]
mu_y = 92.2 # initial depressed vertical tune [deg]
bunch_kind = 'kv' 

# Space charge solver
max_solver_spacing = 0.1 # [m]
min_solver_spacing = 1e-6 # [m]
gridpts = (128, 128, 1) # (x, y, z)

# Set initial depressed tunes
lattice = hf.fodo_lattice(tunes_x[0], tunes_y[0], cell_length, fill_fac=0.5, start='quad')
matcher = Matcher(lattice, kin_energy, eps_x, eps_y)
perveance = matcher.set_tunes(mu_x, mu_y, verbose=2)
intensity = hf.get_intensity(perveance, mass, kin_energy, bunch_length)
print matcher.tunes()

# Create initial bunch
kws = dict()
if bunch_kind == 'gaussian':
    kws['cut_off'] = 3.0
bunch, params_dict = hf.coasting_beam(bunch_kind, n_parts, matcher.twiss(), (eps_x, eps_y), 
                                      bunch_length, mass, kin_energy, intensity, **kws)

# Track bunch
coords, moments = [], []
def measure(bunch):
    X = get_coords(bunch)
    Sigma = np.cov(X.T)
    coords.append(X)
    moments.append(Sigma[np.triu_indices(4)])

measure(bunch)
for mu_x0, mu_y0 in tqdm(zip(tunes_x, tunes_y)):
    lattice = hf.fodo_lattice(mu_x0, mu_y0, cell_length, fill_fac=0.5, start='quad')
    lattice.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*gridpts)
    sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)
    lattice.trackBunch(bunch, params_dict)
    measure(bunch)
    
np.save('data/coords.npy', coords)
np.save('data/moments.npy', moments)