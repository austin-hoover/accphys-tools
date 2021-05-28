"""
This script optimizes the injection kickers to get a certain position
and slope at the foil. 

Call the last kicker on the right B and the second-to-last kicker A. The final 
beam coordinates must be y = y' = 0. Thus, the beam must have y = 0 when it hits
B since all B can do is increase y' to zero.
 
So the job of A is to change y' such that y = 0 at B. But if A can only decrease
y', its job is impossible if y' is too negative when it hits A. So there is a 
limit to the slope we can have at the foil *if the kickers are unipolar*. If
the are bipolar, then there is more freedom.
"""
import sys
import numpy as np
import scipy.optimize as opt

from bunch import Bunch
from orbit.analysis import AnalysisNode, add_analysis_node, add_analysis_nodes
from orbit.teapot import teapot, TEAPOT_Lattice, DriftTEAPOT
from orbit.time_dep import time_dep
from orbit.utils import helper_funcs as hf

from helpers import get_traj, get_part_coords, track_part

sys.path.append('/Users/46h/Research/code/accphys/tools')
from utils import delete_files_not_folders
delete_files_not_folders('_output/')


# Initial settings
#------------------------------------------------------------------------------
x_foil = 0.0492 
y_foil = 0.0468
kin_energy = 0.800 # [GeV]
mass = 0.93827231 # [GeV/c^2]
turns = 1000
macros_per_turn = 260
intensity = 1.5e14

# Initial and final coordinates at s = 0
inj_coords_t0 = np.array([x_foil, 0.0, y_foil, 0.0])
inj_coords_t1 = np.array([x_foil - 0.030, 0.0, y_foil, -0.001])


# Lattice setup
#------------------------------------------------------------------------------
latfile = '_latfiles/SNSring_noRF_nux6.18_nuy6.18.lat'
latseq = 'rnginj'
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX(latfile, latseq)
ring.initialize()
ring.set_fringe(False)
ring_length = ring.getLength()


# Injection kickers
#------------------------------------------------------------------------------
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']
kicker_param_names = ['kx', 'ky', 'kx', 'ky', 'ky', 'kx', 'ky', 'kx']
kicker_nodes = [ring.getNodeForName(name) for name in kicker_names]

# Maximum injection kicker angles at 1 GeV kinetic energy [mrad]
min_kicker_angles = 1.15 * np.array([0.0, 0.0, -7.13, -7.13, -7.13, -7.13, 0.0, 0.0])
max_kicker_angles = 1.15 * np.array([12.84, 12.84, 0.0, 0.0, 0.0, 0.0, 12.84, 12.84])

# Scale angles based on actual kinetic energy
scale_factor = hf.get_pc(mass, 1.0) / hf.get_pc(mass, kin_energy)
min_kicker_angles *= scale_factor
max_kicker_angles *= scale_factor

# Convert from mrad to rad
min_kicker_angles *= 1e-3
max_kicker_angles *= 1e-3

# ARTIFICIALLY INCREASE KICKER LIMITS. Values seem to be less than the defaults.
artificial_increase_factor = 1.5
min_kicker_angles *= artificial_increase_factor
max_kicker_angles *= artificial_increase_factor

def set_kicker_angles(angles, region='all'):
    """Set kicker angles in one half of the injection region."""
    if region == 'left':
        lo, hi = 0, 4
    elif region == 'right':
        lo, hi = 4, 8
    elif region == 'all':
        lo, hi = 0, 8
    for node, param_name, angle in zip(kicker_nodes[lo:hi], kicker_param_names[lo:hi], angles):
        node.setParam(param_name, angle)
        
def get_kicker_angles():
    return [node.getParam(param) for node, param in zip(kicker_nodes, kicker_param_names)]

def optimize_kickers(inj_coords, **kws):
    """Ensure closed orbit at s = 0 has [x, x', y, y'] = inj_coords.""" 
    kicker_angles = []
    for region in ['right', 'left']:
        if region == 'right':
            sublattice = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
            lb = min_kicker_angles[4:]
            ub = max_kicker_angles[4:]
        elif region == 'left':
            sublattice = hf.get_sublattice(ring, 'inj_start', None)
            sublattice.reverseOrder() # track backwards from s = 0 
            inj_coords[[1, 3]] *= -1 # initial slopes therefore change sign
            lb = min_kicker_angles[:4]
            ub = max_kicker_angles[:4]
            
        def penalty(angles):
            set_kicker_angles(angles, region)
            coords_outside_inj = track_part(sublattice, inj_coords, mass, kin_energy)
            return 1e6 * sum(np.square(coords_outside_inj))
        
        guess = np.zeros(4)
        result = opt.least_squares(penalty, guess, bounds=(lb, ub), **kws)
        kicker_angles[:0] = result.x    
    return np.array(kicker_angles)

print 'Optimizing injection kickers.'
kws = dict(max_nfev=10000, verbose=1)
kicker_angles_t0 = optimize_kickers(inj_coords_t0, **kws)  
kicker_angles_t1 = optimize_kickers(inj_coords_t1, **kws)  

# Save initial/final closed orbit trajectory 
ring.split(0.01)
inj_region1 = hf.get_sublattice(ring, 'inj_start', None)
inj_region2 = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
for i, kicker_angles in enumerate([kicker_angles_t0, kicker_angles_t1]):
    set_kicker_angles(kicker_angles)
    coords1, positions1 = get_traj(inj_region1, [0, 0, 0, 0], mass, kin_energy)
    coords2, positions2 = get_traj(inj_region2, 1e-3 * coords1[-1], mass, kin_energy)
    coords = np.vstack([coords1, coords2])
    positions = np.hstack([positions1, positions2 + positions1[-1]])
    np.save('_output/data/inj_region_coords_t{}.npy'.format(i), coords)
    np.save('_output/data/inj_region_positions_t{}.npy'.format(i), positions)
    np.savetxt('_output/data/kicker_angles_t{}.dat'.format(i), kicker_angles)