"""
This script simulates SNS production painting without any nonlinear 
effects and without space charge. 
"""

import math
import sys
import numpy as np
import scipy.optimize as opt
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


# Initial settings
#------------------------------------------------------------------------------
x_foil = 0.0492 
y_foil = 0.0468
kin_energy = 0.800 # [GeV]
mass = 0.93827231 # [GeV/c^2]
intensity = 1.5e14
turns = 1000.0
macros_per_turn = 260
space_charge_calc = False

# Initial and final coordinates at s = 0
coords_s0_t0 = np.array([x_foil - 0.000, 0.000, y_foil - 0.000, 0.000])
coords_s0_t1 = np.array([x_foil - 0.030, 0.000, y_foil - 0.000, 0.003])

# Load SNS ring
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX('_latfiles/SNSring_linear_noRF_nux6.18_nuy6.18.lat', 'rnginj')
ring.set_fringe(False)
ring.initialize()
ring_length = ring.getLength()


# Injection kicker magnet specs
#------------------------------------------------------------------------------
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']
param_names = ['kx', 'ky', 'kx', 'ky', 'ky', 'kx', 'ky', 'kx']
kicker_nodes = [ring.getNodeForName(name) for name in kicker_names]

# Maximum kicker angles for 1 GeV kinetic energy [mrad]
max_kicker_angles = 1.15 * np.array([12.84, 12.84, 7.12, 7.12, 
                                     7.12, 7.12, 12.84, 12.84])

# Scale max angles based on kinetic energy and convert to radians
def get_momentum(kin_energy, mass):
    return np.sqrt(kin_energy * (kin_energy + 2*mass))

max_kicker_angles *= (get_momentum(1.0, mass) / get_momentum(kin_energy, mass))
max_kicker_angles *= 1e-3

# Scale even larger so that it will work
factor = 10
max_kicker_angles *= 10
print 'ALERT: kicker limits increased by factor of {}!'.format(factor)


# Beam set up
#------------------------------------------------------------------------------
# Initialize bunch_
bunch = Bunch()
bunch.mass(mass)
bunch.macroSize(intensity / turns / macros_per_turn)
bunch.getSyncParticle().kinEnergy(kin_energy)
lostbunch = Bunch()
lostbunch.addPartAttr('LostParticleAttributes')
params_dict = {'bunch':bunch, 'lostbunch':lostbunch}
sync_part = bunch.getSyncParticle()

# Transverse linac distribution
order = 3.0
alphax = 0.063
alphay = 0.063
betax = 10.209 
betay = 10.776
emitlim = 0.152 * 2 * (order + 1) * 1e-6
emitlim *= 1.0
xcenterpos = x_foil
ycenterpos = y_foil
xcentermom = ycentermom = 0.0

# Longitudinal linac distribution
zlim = (120.0 / 360.0) * ring_length
zmin, zmax = -zlim, zlim
eoffset = 0.0
deltaEfrac = 0.0

dist_x = JohoTransverse(order, alphax, betax, emitlim, xcenterpos, xcentermom)
dist_y = JohoTransverse(order, alphay, betay, emitlim, ycenterpos, ycentermom)
dist_z = UniformLongDist(zmin, zmax, sync_part, eoffset, deltaEfrac)


# Configure injection kickers
#------------------------------------------------------------------------------
def set_kicker_angles(angles, region='all'):
    """Set kicker angles in one half of the injection region."""
    if region == 1:
        lo, hi = (0, 4)
    elif region == 2:
        lo, hi = (4, 8)
    elif region == 'all':
        lo, hi = (0, 8)
    for node, param_name, angle in zip(kicker_nodes[lo:hi], param_names[lo:hi], angles):
        node.setParam(param_name, angle)
        
def get_part_coords(bunch_):
    """Return list of transverse particle coordinates from bunch_."""
    return [bunch_.x(0), bunch_.xp(0), bunch_.y(0), bunch_.yp(0)]
        
def track_part(lattice, init_coords):
    """Return coords after tracking single particle through lattice."""
    bunch_, params_dict_ = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch_.addParticle(x, xp, y, yp, 0.0, 0.0)
    lattice.trackBunch(bunch_, params_dict_)
    return get_part_coords(bunch_)

def set_inj_region_closed_orbit(coords_s0, **kws):
    """Ensure closed orbit at s = 0 has [x, x', y, y'] = coords_s0.""" 
    kicker_angles = []
    for region in [2, 1]:
        if region == 2:
            sublattice = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
            bounds = (-max_kicker_angles[4:], max_kicker_angles[4:])
        elif region == 1:
            sublattice = hf.get_sublattice(ring, 'inj_start', None)
            sublattice.reverseOrder() # track backwards from s = 0 to start of injection region
            coords_s0[[1, 3]] *= -1 # initial slopes therefore change sign
            bounds = (-max_kicker_angles[:4], max_kicker_angles[:4])
            
        def penalty(angles):
            """Penalize nonzero coordinates outside injection region."""
            set_kicker_angles(angles, region)
            return 1e6 * sum(np.square(track_part(sublattice, coords_s0)))
        
#         result = opt.least_squares(penalty, np.zeros(4), bounds=bounds, **kws)
        result = opt.minimize(penalty, np.zeros(4), bounds=opt.Bounds(*bounds),
                              method='trust-constr', options=kws)
        kicker_angles[:0] = result.x
    return np.array(kicker_angles)

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
    field_in_tesla_per_meter = kq * hf.get_Brho(mass, kin_energy)
    quad_node.setParam('kq', 1.0 * kq)

# Optimize kicker angles to get the desired coordinates at the foil
print 'Finding correct initial kicker strengths.'
kicker_angles_t0 = set_inj_region_closed_orbit(coords_s0_t0, verbose=2)  
print 'Finding correct final kicker strengths.'
kicker_angles_t1 = set_inj_region_closed_orbit(coords_s0_t1, verbose=2)  
set_kicker_angles(kicker_angles_t0)

# # Add time-dependence to kicker nodes
ring.setLatticeOrder()
t0 = 0.000 # [s]
t1 = 0.001 # [s]
amps_t0 = np.ones(8)
amps_t1 = kicker_angles_t1 / kicker_angles_t0
for node, amp_t0, amp_t1 in zip(kicker_nodes, amps_t0, amps_t1):
    waveform = SquareRootWaveform(sync_part, t0, t1, amp_t0, amp_t1)
    ring.setTimeDepNode(node.getParam('TPName'), waveform)
    
    
# Add space charge nodes
#------------------------------------------------------------------------------
# In the future, use calculation from examples repository with cylindrical 
# conduction boundary and longitudinal space charge node. Right now we assume
# no such boundary and no longitudinal dynamics.
grid_size = (128, 128, 1)
max_solver_spacing = 0.2
min_solver_spacing = 1e-6

if space_charge_calc:
    ring.split(max_solver_spacing)    
    calc2p5d = SpaceChargeCalc2p5D(*grid_size)
    scLatticeModifications.setSC2p5DAccNodes(ring, min_solver_spacing, calc2p5d)

    
# Perform injection painting 
#------------------------------------------------------------------------------
foil_xmin = xcenterpos - 0.0085
foil_xmax = xcenterpos + 0.0085
foil_ymin = ycenterpos - 0.0080
foil_ymax = ycenterpos + 0.100
foil_boundaries = [foil_xmin, foil_xmax, foil_ymin, foil_ymax]
injection_node = TeapotInjectionNode(macros_per_turn, bunch, lostbunch, 
                                     foil_boundaries, dist_x, dist_y, dist_z)
addTeapotInjectionNode(ring, 0.0, injection_node)


bunch_monitor_node = AnalysisNode(0.0, 'bunch_monitor')
injection_node.addChildNode(bunch_monitor_node, injection_node.EXIT)
print 'Tracking'
for _ in trange(1000):
    ring.trackBunch(bunch, params_dict)
coords = bunch_monitor_node.get_data('bunch_coords', 'all_turns')
for i, X in enumerate(coords):
    np.save('_output/data/X_{}.npy'.format(i), X)


# Save injection region closed orbit trajectory
#------------------------------------------------------------------------------
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

ring.split(0.01)
inj_region1 = hf.get_sublattice(ring, 'inj_start', None)
inj_region2 = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
for i, kicker_angles in enumerate([kicker_angles_t0, kicker_angles_t1]):
    set_kicker_angles(kicker_angles)
    coords1, positions1 = get_traj(inj_region1, [0, 0, 0, 0])
    coords2, positions2 = get_traj(inj_region2, 1e-3 * coords1[-1])
    coords = np.vstack([coords1, coords2])
    positions = np.hstack([positions1, positions2 + positions1[-1]])
    np.save('_output/data/inj_region_coords_t{}.npy'.format(i), coords)
    np.save('_output/data/inj_region_positions_t{}.npy'.format(i), positions)
    
np.savetxt('_output/data/kicker_angles_t0.dat', kicker_angles_t0)
np.savetxt('_output/data/kicker_angles_t1.dat', kicker_angles_t1)