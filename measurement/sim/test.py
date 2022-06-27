"Compare the beam evolution in the RTBT with/without space charge."""

from __future__ import print_function
import sys
import os

from tqdm import tqdm
from tqdm import trange
import numpy as np

from bunch import Bunch
from orbit_utils import Matrix
from spacecharge import SpaceChargeCalc2p5D
from orbit.diagnostics import analysis
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import WireScannerNode
from orbit.lattice import AccNode
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.utils import helper_funcs as hf
from orbit.utils.general import ancestor_folder_path
from orbit.utils.general import delete_files_not_folders
from orbit.utils.general import load_stacked_arrays



# Settings
#------------------------------------------------------------------------------
beam_input_file = '_input/coords_rtbt.npz'
kin_energy = 0.8 # [GeV]
mass = 0.93827231 # [GeV/c^2]
final_intensity = 5.4e13

turns = list(range(0, 400, 50))
final_turn = turns[-1]

# 1.5e11 is added to the intensities because turn 0 is a single minpulse.
intensities = final_intensity * np.divide(turns, float(final_turn)) + 1.5e11
print('Turns =', turns)
print('Intensities =', intensities)

madx_file = '_input/rtbt.lat'
madx_seq = 'whole1'
init_twiss = {'alpha_x': -0.25897, 
              'alpha_y': 0.9749,
              'beta_x': 2.2991, 
              'beta_y': 14.2583}
ws_names = ['ws20', 'ws21', 'ws23', 'ws24']

# Space charge solver
max_solver_spacing = 1.0 # [m]
min_solver_spacing = 0.00001 # [m]
gridpts = (128, 128, 1)


# Initialization
#------------------------------------------------------------------------------
lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
lattice.set_fringe(False)

rec_node_names = []
for node in lattice.getNodes():
    if 'Drift' in node.getName():
        continue
    rec_node_names.append(node.getName())

# Save node names.
file = open('_output/data/rec_node_names.txt', 'w')
for node_name in rec_node_names:
    file.write(node_name + '\n')
file.close()

# Save node positions.
rec_node_positions = []
node_pos_dict = lattice.getNodePositionsDict()
for node_name in rec_node_names:
    node = lattice.getNodeForName(node_name)
    pos_start, pos_stop = node_pos_dict[node]
    rec_node_positions.append([pos_start, pos_stop])
np.savetxt('_output/data/rec_node_positions.dat', rec_node_positions)
    
# Load the bunch coordinates.
print('Loading initial beam coordinates.')
coords = None
if '.dat' in beam_input_file or '.txt' in beam_input_file:
    X0 = np.loadtxt(beam_input_file)
elif '.npy' in beam_input_file:
    X0 = np.load(beam_input_file)
elif '.npz' in beam_input_file:
    coords = load_stacked_arrays(beam_input_file)
    for t in range(len(coords)):
        coords[t][:, :4] *= 0.001
if coords is None:
    coords = [X0]
        
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)
params_dict = {'bunch': bunch} 

def set_bunch_coords(bunch, X):
    if bunch.getSize() == X.shape[0]:
        for i, (x, xp, y, yp, z, dE) in enumerate(X):
            bunch.x(i, x)
            bunch.xp(i, xp)
            bunch.y(i, y)
            bunch.yp(i, yp)
            bunch.z(i, z)
            bunch.dE(i, dE)
    else:
        bunch.deleteAllParticles()
        for (x, xp, y, yp, z, dE) in X:
            bunch.addParticle(x, xp, y, yp, z, dE)

        
for space_charge in [True, False]:
    
    lattice = TEAPOT_Lattice()
    lattice.readMADX(madx_file, madx_seq)
    lattice.set_fringe(False)

    lattice.split(max_solver_spacing)    
    if space_charge:
        calc2p5d = SpaceChargeCalc2p5D(*gridpts)
        sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

    bunch_stats_nodes = dict()
    for node_name in rec_node_names:
        bunch_stats_node = BunchStatsNode(mm_mrad=True)
        node = lattice.getNodeForName(node_name)
        node.addChildNode(bunch_stats_node, node.ENTRANCE)
        bunch_stats_nodes[node_name] = bunch_stats_node
        
    frame = -1
    
    X = coords[frame]
    set_bunch_coords(bunch, X)
    intensity = intensities[frame]
    turn = turns[frame]
    macro_size = max(1, int(intensity / bunch.getSize()))
    bunch.macroSize(macro_size)

    lattice.trackBunch(bunch, params_dict)
    Sigmas = []
    for node_name in rec_node_names:
        bunch_stats_node = bunch_stats_nodes[node_name]
        bunch_stats = bunch_stats_node.get_data(-1)
        Sigmas.append(bunch_stats.Sigma)

    np.save('_output/data/Sigmas_{}_sc{}.npy'.format(frame, space_charge), Sigmas)
    
np.savetxt('_output/data/turns.dat', turns)
np.savetxt('_output/data/intensities.dat', intensities)