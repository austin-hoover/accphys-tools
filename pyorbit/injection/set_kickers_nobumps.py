"""
This script simulates SNS production painting without any nonlinear 
effects and without space charge. 
"""

import math
import sys
import numpy as np
import scipy.optimize as opt
from scipy.constants import speed_of_light
from tqdm import tqdm, trange

from bunch import Bunch
from orbit.analysis import AnalysisNode, add_analysis_node, add_analysis_nodes
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.time_dep import time_dep
from orbit.teapot import teapot, TEAPOT_Lattice, DriftTEAPOT
from spacecharge import Boundary2D
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from spacecharge import SpaceChargeCalc2p5D, Boundary2D
from spacecharge import LSpaceChargeCalc
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.utils import helper_funcs as hf

sys.path.append('/Users/46h/Research/code/accphys/tools')
from utils import delete_files_not_folders
delete_files_not_folders('_output/')


# Settings
x_foil = 0.0492 
y_foil = 0.0468
kin_energy = 0.800 # [GeV]
mass = 0.93827231 # [GeV/c^2]

# Create lattice
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX('_latfiles/SNSring_linear_noRF_nux6.18_nuy6.18.lat', 'rnginj')
ring.initialize()
ring.set_fringe(False)

# Here we increase the strengths of the quadrupoles in the injection
# region. They are a bit lower than the default values; because of this, the 
# kickers can't get the required slope at s = 0. We might have to fix the 
# strength of these quads, then set the tunes equal to each other by varying
# the other quads in the ring. I think this must be done in MADX. Put this 
# on hold for now.
quad_names = ['qth_a10', 'qtv_a11', 'qtv_a12', 'qth_a13']
for quad_name in quad_names:
    quad_node = ring.getNodeForName(quad_name)
    kq = quad_node.getParam('kq')
    print quad_name, kq * hf.get_Brho(mass, kin_energy)
    quad_node.setParam('kq', 1.0 * kq)
    
# Injection kicker magnets (listed in order)
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']
param_names = ['kx', 'ky', 'kx', 'ky', 'ky', 'kx', 'ky', 'kx']
kicker_nodes = [ring.getNodeForName(name) for name in kicker_names]

# Maximum injection kicker angles for 1 GeV kinetic energy [mrad]
max_kick_angles = np.array([12.84, 12.84, 7.13, 7.12, 7.12, 7.12, 12.84, 12.84])

# Those are the old limits -- new limits are 15% higher
max_kick_angles *= 1.15

# Scale angles based on actual kinetic energy
def get_momentum(kin_energy, mass):
    return np.sqrt(kin_energy * (kin_energy + 2*mass))
max_kick_angles *= (get_momentum(1.0, mass) / get_momentum(kin_energy, mass))

# Convert from mrad to rad
max_kick_angles *= 1e-3


def set_kick_angles(angles, region=1):
    """Set kicker angles in one half of the injection region."""
    lo, hi = (0, 4) if region == 1 else (4, 8)
    for node, param_name, angle in zip(kicker_nodes[lo:hi], param_names[lo:hi], angles):
        node.setParam(param_name, angle)
    
                
def get_part_coords(bunch):
    """Return list of transverse particle coordinates from bunch."""
    return [bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)]
        

def track_part(lattice, init_coords):
    """Return coords after tracking single particle through lattice."""
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
    lattice.trackBunch(bunch, params_dict)
    return get_part_coords(bunch)


def set_inj_region_closed_orbit(coords_s0, **kws):
    """Ensure closed orbit at s = 0 has [x, x', y, y'] = coords_s0.""" 
    kick_angles = []
    for region in [2, 1]:
        if region == 2:
            sublattice = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
            bounds = (-max_kick_angles[4:], max_kick_angles[4:])
        elif region == 1:
            sublattice = hf.get_sublattice(ring, 'inj_start', None)
            sublattice.reverseOrder() # track backwards from s = 0 to start of injection region
            coords_s0[[1, 3]] *= -1 # initial slopes therefore change sign
            bounds = (-max_kick_angles[:4], max_kick_angles[:4])
            
        def penalty(angles):
            """Penalize nonzero coordinates outside injection region."""
            set_kick_angles(angles, region)
            return 1e6 * sum(np.square(track_part(sublattice, coords_s0)))
        
        result = opt.least_squares(penalty, np.zeros(4), bounds=bounds, **kws)
        kick_angles[:0] = result.x
    return kick_angles
    
    
# Get necessary kicker angles
coords_s0 = np.array([x_foil - 0.010, 0.000, y_foil, -0.001])
kick_angles = set_inj_region_closed_orbit(coords_s0, verbose=1)  

print 'Constraint violations:'
diffs = np.abs(kick_angles) - max_kick_angles
for name, diff in zip(kicker_names, diffs):
    print '{} | {}'.format(name, 1000 * hf.step_func(diff))
    
def get_traj(lattice, init_coords):
    """Return single particle trajectory through lattice."""
    bunch_, params_dict_ = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch_.addParticle(x, xp, y, yp, 0.0, 0.0)
    monitors = add_analysis_nodes(lattice, kind='bunch_monitor')
    lattice.trackBunch(bunch_, params_dict_)
    coords, positions = [], []
    for node in monitors:
        coords.append(node.get_data('bunch_coords')[0])
        positions.append(node.position)
        node.clear_data()
    return np.array(coords), np.array(positions)

# Save injection region trajectory
ring.split(0.01)
inj_region1 = hf.get_sublattice(ring, 'inj_start', None)
inj_region2 = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
coords1, positions1 = get_traj(inj_region1, [0, 0, 0, 0])
coords2, positions2 = get_traj(inj_region2, 1e-3 * coords1[-1])
coords = np.vstack([coords1, coords2])
positions = np.hstack([positions1, positions2 + positions1[-1]])
np.save('_output/data/coords.npy', coords)
np.save('_output/data/positions.npy', positions)