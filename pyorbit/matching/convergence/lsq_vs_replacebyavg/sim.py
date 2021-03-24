"""
For a range of beam perveances, this script runs both the scipy least squares
optimizer and the 'homemade' optimizer to find the matched envelope and saves
the final value of the cost function. 

In a FODO lattice with skew quadrupoles, the matched solution in the bare
lattice is a diagonal line in x-y space. If space charge is small but nonzero,
the matched solution is approximately a diagonal line with an opposite tilt 
angle. The packaged optimizer struggles with this case, but the 
'replace by average' method is able to find the matched beam (not always).
"""

# Standard 
import sys
import copy
# Third party
import numpy as np
from tqdm import tqdm, trange
# PyORBIT
from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import (
    AnalysisNode,
    add_analysis_nodes, 
    get_analysis_nodes_data, 
    clear_analysis_nodes_data)
from orbit.analysis.analysis import covmat2vec
from orbit.coupling import bogacz_lebedev as BL
from orbit.envelope import Envelope
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.space_charge.envelope import set_env_solver_nodes, set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys') 
from tools.utils import delete_files_not_folders
    
    
# Settings
#------------------------------------------------------------------------------

# General
mass = 0.93827231 # GeV/c^2
energy = 1.0 # GeV/c^2

# Lattice
latfile = '_latfiles/fodo_skew_quadstart.lat'
latseq = 'fodo'
fringe = False

# Initial beam
mode = 1
eps = 50e-6 # intrinsic emitance
ex_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.05 # [m]
min_solver_spacing = 1e-6

# Matching
match = True 
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0. # if nonzero, perturb the matched envelope
method = 'replace_by_avg'

delete_files_not_folders('_output/')


#------------------------------------------------------------------------------

intensities = np.linspace(0, 1.5e14, 75)
env = Envelope(eps, mode, ex_frac, mass, energy, length=5.0)

def cost_func(env, lattice):
    residuals = env._mismatch_error(lattice)
    return 0.5 * np.sum(residuals**2)

cost_lists = []
matched_params_lists = []
tbt_params_lists = []

for method in ('lsq', 'replace_by_avg'):
    print 'Method =', method
    cost_list = []
    tbt_params_list = [] 
    for intensity in tqdm(intensities):
        lattice = hf.lattice_from_file(latfile, latseq, fringe)
        env.match_bare(lattice)
        env.set_spacecharge(intensity)
        solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
        if intensity > 0:
            env.match(lattice, solver_nodes, method=method, tol=tol, verbose=0)
        cost_list.append(cost_func(env, lattice))
        tbt_params_list.append(env.track_store_params(lattice, 20))
    cost_lists.append(cost_list)
    tbt_params_lists.append(tbt_params_list)
    
np.save('_output/data/cost_lists.npy', cost_lists)
np.save('_output/data/tbt_params_lists.npy', tbt_params_lists)
np.save('_output/data/perveances.npy', 
        hf.get_perveance(mass, energy, intensities/lattice.getLength()))