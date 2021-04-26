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
from orbit.utils.orbit_mpi_utils import bunch_orbit_to_pyorbit
from orbit.utils.orbit_mpi_utils import bunch_pyorbit_to_orbit

from orbit.analysis import AnalysisNode, add_analysis_node, add_analysis_nodes
from orbit.utils import helper_funcs as hf

from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot, TEAPOT_Lattice, DriftTEAPOT

from orbit.time_dep import time_dep
from orbit.time_dep.waveforms import SquareRootWaveform, ConstantWaveform

from orbit.injection import TeapotInjectionNode, addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import JohoTransverse, JohoLongitudinal, SNSESpreadDist
from orbit.injection import UniformLongDist

from spacecharge import Boundary2D
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from spacecharge import SpaceChargeCalc2p5D, Boundary2D
from spacecharge import LSpaceChargeCalc
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode

sys.path.append('/Users/46h/Research/code/accphys/tools')
from utils import delete_files_not_folders
delete_files_not_folders('_output/')


# Settings
x_foil = 0.0492 
y_foil = 0.0468
kin_energy = 0.850 # [GeV]
mass = 0.93827231 # [GeV/c^2]
intensity = 1.5e14
turns = 1000.0
macros_per_turn = 260
space_charge_calc = False


# Determine correct kicker settings
#------------------------------------------------------------------------------
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX('_latfiles/SNSring_linear_noRF_nux6.18_nuy6.18.lat', 'rnginj')
ring.initialize()
ring.set_fringe(False)

# Injection kicker magnets (listed in order)
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickh_a12', 'ikickv_a12', 'ikickh_a13', 'ikickv_a13']
kicker_nodes = [ring.getNodeForName(name) for name in kicker_names]

# Orbit correctors available in the injection region (listed in order). 
# Correctors outside injection kickers ('dmcv') give only vertical kicks,
# while correctors inside injection kickers ('dchv') give both horizontal 
# and vertical kicks.
orbit_corrector_names = ['dmcv_a09', 'dchv_a10', 'dchv_a13', 'dmcv_b01']
orbit_corrector_nodes = [ring.getNodeForName(name) for name in orbit_corrector_names]
orbit_corrector_nodes_v = orbit_corrector_nodes
orbit_corrector_nodes_h = [orbit_corrector_nodes[0], orbit_corrector_nodes[-1]]

# Maximum injection kicker angles for 1 GeV kinetic energy [mrad]
max_angles = np.array([12.84, 12.84, 7.13, 7.13, 7.13, 7.94, 12.84, 12.84])

# Add bounds for orbit corrector magnets (currently unknown). The first four
# elements are for the vertical strengths, and the second four elements are 
# for the two horizontal strengths.
max_co_angle = 1e-3
max_angles_closed_orbit_correctors_v = np.array(4 * [max_co_angle])
max_angles_closed_orbit_correctors_h = np.array(2 * [max_co_angle])

# Scale bounds based on actual kintetic energy
def get_momentum(kin_energy, mass):
    return np.sqrt(kin_energy * (kin_energy + 2*mass))
max_angles *= get_momentum(1.0, mass) / get_momentum(kin_energy, mass)

# Convert angles from mrad to rad
max_angles *= 5e-3


def set_kicker_strengths(kicker_nodes, kicker_strengths):
    for i, (node, strength) in enumerate(zip(kicker_nodes, kicker_strengths)):
        node.setParam('kx' if i % 2 == 0 else 'ky', strength)
        
            
def get_part_coords(bunch):
    return [bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)]
        

def track_part(lattice, init_coords):
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
    lattice.trackBunch(bunch, params_dict)
    return get_part_coords(bunch)


def set_inj_region_closed_orbit(coords_s0, max_nfev=int(1e4), verbose=2):
    _ikick_strengths = []
    _cokick_strengths = []
    for region in ['after', 'before']:
        if region == 'after':
            sublattice = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
            kicker_nodes_sublist = kicker_nodes[4:]
            orbit_correct_nodes_v_sublist = orbit_corrector_nodes_v[2:]
            orbit_correct_nodes_h_sublist = orbit_corrector_nodes_h[1:]
            
        elif region == 'before':
            sublattice = hf.get_sublattice(ring, 'inj_start', None)
            sublattice.reverseOrder() # track backwards from s = 0 to start of injection region
            coords_s0[1] *= -1 # initial slopes therefore change sign
            coords_s0[3] *= -1
            kicker_nodes_sublist = kicker_nodes[:4]
            orbit_correct_nodes_v_sublist = orbit_corrector_nodes_v[:2]
            orbit_correct_nodes_h_sublist = orbit_corrector_nodes_h[:1]
            
        def penalty(kick_strengths):
            """Penalize nonzero closed orbit coordinates outside injection region.
            
            The first four elements of `kick_strengths` correspond to the injection
            kicker magnets in this half of the injection region. 
            
            The next two elements correspond to the vertical orbit correctors in 
            this half of the injection region. 
            
            The final element corresponds to the horizontal orbit corrector in 
            this half of the injection region. I don't know why there aren't equal 
            numbers of horizontal and vertical correctors.
            """
            kicker_strengths = kick_strengths[:4]
            orbit_corrector_strengths = kick_strengths[4:]
            set_kicker_strengths(kicker_nodes_sublist, kicker_strengths)
            for node, strength in zip(orbit_correct_nodes_v_sublist, orbit_corrector_strengths[:2]):
                node.setParam('ky', strength)
            for node, strength in zip(orbit_correct_nodes_h_sublist, orbit_corrector_strengths[2:]):
                node.setParam('kx', strength)
            return 1e6 * sum(np.square(track_part(sublattice, coords_s0)))
    
        if region == 'after':
            ub = max_angles[4:]
            ub = np.append(ub, max_angles_closed_orbit_correctors_v[2:])
            ub = np.append(ub, max_angles_closed_orbit_correctors_h[1:])
        if region == 'before':
            ub = max_angles[:4]
            ub = np.append(ub, max_angles_closed_orbit_correctors_v[:2])
            ub = np.append(ub, max_angles_closed_orbit_correctors_h[:1])
        bounds = (-ub, ub)
#         bounds = (-np.inf, np.inf)
        result = opt.least_squares(penalty, np.zeros(7), bounds=bounds,
                                   max_nfev=max_nfev, verbose=verbose)
        _ikick_strengths.extend(result.x[:4])
        _cokick_strengths.extend(result.x[4:])
    return _ikick_strengths, _cokick_strengths

        
def get_traj(lattice, init_coords):
    """Return single particle trajectory through lattice."""
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
    monitors = add_analysis_nodes(lattice, kind='bunch_monitor')
    lattice.trackBunch(bunch, params_dict)
    coords, positions = [], []
    for node in monitors:
        coords.append(node.get_data('bunch_coords')[0])
        positions.append(node.position)
        node.clear_data()
    return np.array(coords), np.array(positions)

    
    
# Get kicker strengths
dx = 0.0
dy = 0.0
dxp = 0.0
dyp = 0.0
coords_s0 = np.array([x_foil + dx, dxp, y_foil + dy, dyp])
_ikick_strengths, _cokick_strengths = set_inj_region_closed_orbit(coords_s0)   

print np.abs(_ikick_strengths) > max_angles
print np.abs(_cokick_strengths) > max_co_angle


# set_kicker_strengths(kicker_nodes, np.zeros(8))


# Save injection region trajectory
inj_region1 = hf.get_sublattice(ring, 'inj_start', None)
inj_region2 = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
coords1, positions1 = get_traj(inj_region1, [0, 0, 0, 0])
coords2, positions2 = get_traj(inj_region2, 1e-3 * coords1[-1])
coords = np.vstack([coords1, coords2])
positions = np.hstack([positions1, positions2 + positions1[-1]])
np.save('_output/data/coords.npy', coords)
np.save('_output/data/positions.npy', positions)
