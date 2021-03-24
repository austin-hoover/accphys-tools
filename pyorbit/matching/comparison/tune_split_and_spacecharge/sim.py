"""
This script compares the matched emittances for a beam in a FODO lattice 
with unequal tunes as the space charge strength is increased.
"""

# Standard 
import sys
# Third party
import numpy as np
from tqdm import tqdm
# PyORBIT
from bunch import Bunch
from orbit.analysis import add_analysis_nodes, get_analysis_nodes_data
from orbit.coupling import bogacz_lebedev as BL
from orbit.envelope import Envelope
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------

# General
mass = 0.93827231 # GeV/c^2
energy = 1.0 # GeV
intensities = np.linspace(0, 5e14, 25)

# Lattice
mux = 80 # total x phase advance [deg]
muy_list = [70, 75, 80, 85, 90] # total y phase advance [deg]
length = 5.0 # lattice length [m]
fill_fac = 0.5 # fraction of lattice occupied by quadrupoles
angle = 0.0 # tilt angle of quadrupoles [deg]
start = 'quad' # whether the lattice entrance is at the center of the quad or drift
fringe = False # whether to include fringe fields

# Initial beam
mode = 1
eps = 50e-6 # intrinsic emitance
ex_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Matching
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}

# Space charge solver
max_solver_spacing = 0.01 # [m]

delete_files_not_folders('_output')

# Simulation
#------------------------------------------------------------------------------

perveances = hf.get_perveance(mass, energy, intensities/length)
env = Envelope(eps, mode, ex_frac, mass, energy, length)
matched_emittance_fracs_list, tune_splits = [], []

for muy in muy_list:
    tune_splits.append(muy - mux)
    print('muy - mux = {:.2f} deg'.format(tune_splits[-1]))
    matched_emittance_fracs = []
    for Q in tqdm(perveances):
        lattice = hf.fodo_lattice(mux, muy, length, fill_fac, angle, start)
        solver_nodes = set_env_solver_nodes(lattice, Q, max_solver_spacing)
        env.perveance = Q
        if Q == 0:
            env.match_bare(lattice)
        else:
            env.match(lattice, solver_nodes, verbose=0)
        matched_emittance_fracs.append(env.emittances() / eps)
    matched_emittance_fracs_list.append(matched_emittance_fracs)

np.savetxt('_output/data/perveances.dat', perveances)
np.savetxt('_output/data/tune_splits.dat', tune_splits)
np.save('_output/data/matched_emittance_fracs_list', matched_emittance_fracs_list)

