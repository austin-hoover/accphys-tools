"""
For a range of beam perveances, this script runs both the scipy least squares
optimizer and the 'homemade' optimizer to find the matched envelope and saves
the final value of the cost function. 

In a FODO lattice with skew quadrupoles, the matched solution in the bare
lattice is a diagonal line in x-y space. If space charge is small but nonzero,
the matched solution is approximately a diagonal line with an opposite tilt 
angle. The packaged optimizer struggles with this case, but the 
'replace by average' method is able to find the matched beam (not always).

The 'replace by average' method is not guaranteed to converge. Use the
least squares optimizer unless 
"""
import sys
import numpy as np
from tqdm import tqdm, trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode
from orbit.analysis import add_analysis_nodes
from orbit.analysis import get_analysis_nodes_data
from orbit.envelope import DanilovEnvelope
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.envelope import set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf
from orbit.utils.general import delete_files_not_folders
from orbit.utils.consts import mass_proton


# Settings
#------------------------------------------------------------------------------
# Lattice
madx_file = '_input/SNSring_nux6.18_nuy6.18.lat'
madx_seq = 'rnginj'
# madx_seq = 'fodo'
print('madx_file = {}'.format(madx_file))

# Initial beam
mass = mass_proton # [GeV/c^2]
kin_energy = 0.8 # [GeV]
intensities = 1.5e14 * np.linspace(0., 0.8, 8)
bunch_length = (45.0 / 64.0) * 248.0 # [m]
mode = 2 # {1, 2} determines sign of angular momentum
eps_l = 20e-6 # nonzero intrinsic emittance [m rad]
eps_x_frac = 0.5 # eps_x / eps_l
nu = np.radians(90) # x-y phase difference

# Space charge solver
max_solver_spacing = 0.1 # [m]
min_solver_spacing = 1e-6
gridpts = (128, 128, 1) # (x, y, z)

# Matching
tol = 1e-4 # absolute tolerance for cost function
verbose = 2 # {0 (silent), 1 (report once at end), 2 (report at each step)}
perturb_radius = 0. # between 0 (no effect) and 1.0
method = 'replace_by_avg' 
n_turns_track = 10


delete_files_not_folders('_output/')


# Simulation
#------------------------------------------------------------------------------
cost_lists, matched_params_lists, tbt_params_lists = [], [], []
env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, length=bunch_length)
for method in ('lsq', 'replace_avg'):
    print 'Method =', method
    cost_list = []
    tbt_params_list = [] 
    for intensity in tqdm(intensities):
        lattice = TEAPOT_Lattice()
        lattice.readMADX(madx_file, madx_seq)
        lattice.set_fringe(False)
        env.match_bare(lattice)
        env.set_intensity(intensity)
        solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
        if intensity > 0:
            env.match(lattice, solver_nodes, method=method, tol=tol, verbose=verbose)
        cost_list.append(0.5 * np.sum(env._residuals(lattice)**2))
        tbt_params_list.append(env.track_store_params(lattice, n_turns_track))
    cost_lists.append(cost_list)
    tbt_params_lists.append(tbt_params_list)
    
np.save('_output/data/cost_lists.npy', cost_lists)
np.save('_output/data/tbt_params_lists.npy', tbt_params_lists)
np.save('_output/data/perveances.npy', 
        hf.get_perveance(mass, kin_energy, intensities/lattice.getLength()))