"""
This script stores the matched beam parameters as a function of s for each 
lattice, beam mode, and intensity. This is done for following lattices:
* fodo_quadstart.lat
    (1) FODO lattice (QF/2 - O - QD - O - QD/2).
    (2) Same as (1), but split the tunes.
    (3) Same as (1), but tilt the focusing quadrupoles by 3 degrees and the 
        defocusing quadrupoles by -3 degrees.
    (4) Same as (1), but add solenoids between the quadrupoles.
"""
from __future__ import print_function
import sys
import os

import numpy as np
from tqdm import tqdm
from tqdm import trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import DanilovEnvelopeBunchMonitorNode
from orbit.diagnostics import add_analysis_node
from orbit.diagnostics import add_analysis_nodes
from orbit.envelope import DanilovEnvelope
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.lattice import AccActionsContainer
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.envelope import set_perveance
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf
from orbit.utils.consts import mass_proton
from orbit.utils.general import delete_files_not_folders

    
# Settings
#------------------------------------------------------------------------------
# General
mass = mass_proton # [GeV/c^2]
kin_energy = 1.0 # [GeV]
bunch_length = 5.0 # [m]
intensities = np.linspace(0, 10e14, 8)

# Lattice
latfiles = [
    '_latfiles/fodo_quadstart.lat',
    '_latfiles/fodo_split_quadstart.lat',
    '_latfiles/fodo_skew_quadstart.lat',
    '_latfiles/fodo_sol_quadstart.lat',
]
latnames = [
    'fodo',
    'fodo_split',
    'fodo_skew',
    'fodo_sol'
]
latseq = 'fodo'

# Initial beam
eps_l = 40e-6 # intrinsic emitance
eps_x_frac = 0.5 # ex/eps
nu = np.radians(90) # x-y phase difference

# Matching
match = True 
verbose = 2 
method = 'lsq' 
max_solver_spacing = 0.01 # [m]

# Output data locations
files = {
    'positions': '_output/data/positions_{}.npy', 
    'perveances': '_output/data/perveances.npy', 
    'tracked_params_list': '_output/data/tracked_params_list_{}_{}.npy',
    'transfer_mats': '_output/data/transfer_mats_{}_{}.npy',
}
data = {key: None for key in files}


delete_files_not_folders('_output')


#------------------------------------------------------------------------------


for latname, latfile in zip(latnames, latfiles):
    print('lattice = {}'.format(latname))
    for mode in (1, 2):
        print('mode = {}'.format(mode))
        tracked_params_list, transfer_mats = [], []
        env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, bunch_length)
        for intensity in tqdm(intensities):
            env.set_intensity(intensity)
            lattice = TEAPOT_Lattice()
            lattice.readMADX(latfile, latseq)
            lattice.set_fringe(False)
            solver_nodes = set_env_solver_nodes(lattice, env.perveance, max_solver_spacing)
            env.match(lattice, solver_nodes, method=method, verbose=verbose)
            transfer_mats.append(env.transfer_matrix(lattice))
            monitor_nodes = add_analysis_nodes(DanilovEnvelopeBunchMonitorNode, lattice, dense=True)
            env.track(lattice)
            tracked_params_list.append([node.get_data(-1).env_params for node in monitor_nodes])            
        # Save data
        data['positions'] = [node.position for node in monitor_nodes]
        data['perveances'] = hf.get_perveance(mass, kin_energy, intensities / bunch_length)
        data['tracked_params_list'] = tracked_params_list
        data['transfer_mats'] = transfer_mats
        for key in files:
            np.save(files[key].format(latname, mode), data[key])
    print()

file = open('_output/data/latnames.txt', 'w')
for latname in latnames:
    file.write(latname + '\n')