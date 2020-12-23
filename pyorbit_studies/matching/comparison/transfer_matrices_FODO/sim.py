"""
This script calculates the effective transfer matrix for the matched beam as 
a function of intensity. It does so for the following variants of the FODO
lattice:
    * FODO, equal x/y tunes
    * FODO, unequal x/y tunes (change the quad strengths)
    * FODO, skew quadrupoles (tilt quadrupoles opposite directions)
    * FODO, solenoids inserted between quadrupoles
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
intensities = np.linspace(0, 5e14, 8)
start = 'quad'

# Lattice
if start == 'drift':
    latfiles = [
        '_latfiles/fodo_driftstart.lat',
        '_latfiles/fodo_split_driftstart.lat',
        '_latfiles/fodo_skew_driftstart.lat',
        '_latfiles/fodo_sol_driftstart.lat'
    ]
elif start == 'quad':
    latfiles = [
        '_latfiles/fodo_quadstart.lat',
        '_latfiles/fodo_split_quadstart.lat',
        '_latfiles/fodo_skew_quadstart.lat',
        '_latfiles/fodo_sol_quadstart.lat'
    ]
latseq = 'fodo'
fringe = False

# Beam
mode = 2
eps = 50e-6
ex_ratio = 0.5
nu = np.radians(90)

# Space charge solver
max_solver_spacing = 0.01

delete_files_not_folders('_output/')

#------------------------------------------------------------------------------

lattice = hf.lattice_from_file(latfiles[0], latseq, fringe)
env = Envelope(eps, mode, ex_ratio, mass, energy, length=lattice.getLength())

for latfile in latfiles:
    
    print latfile
    transfer_mats = []
    
    for intensity in tqdm(intensities):
        lattice = hf.lattice_from_file(latfile, latseq, fringe)
        env.set_spacecharge(intensity)
        solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
        env.match(lattice, solver_nodes, verbose=0)
        transfer_mats.append(env.transfer_matrix(lattice))
                
    filename = '_output/data/{}.npy'.format(latfile[10:].split('.')[0])
    np.save(filename, transfer_mats)
    
perveances = hf.get_perveance(mass, energy, intensities/lattice.getLength())
np.savetxt('_output/data/perveances.dat', perveances)