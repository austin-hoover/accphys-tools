"""
This script simulates SNS production painting.
"""
import sys
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm, trange
from pprint import pprint

from bunch import Bunch
from foil import Foil
from impedances import LImpedance, TImpedance
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D

from orbit.analysis import AnalysisNode, add_analysis_node, add_analysis_nodes
from orbit.collimation import TeapotCollimatorNode, addTeapotCollimatorNode
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import StatLats, Moments
from orbit.diagnostics import TeapotStatLatsNode
from orbit.diagnostics import TeapotMomentsNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.diagnostics import addTeapotStatLatsNodeSet
from orbit.diagnostics import addTeapotMomentsNodeSet
from orbit.foils import TeapotFoilNode, addTeapotFoilNode
from orbit.impedances import LImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import addImpedanceNode
from orbit.injection import TeapotInjectionNode
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import JohoTransverse, JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.rf_cavities import RFNode, RFLatticeModifications
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from orbit.teapot import teapot, TEAPOT_Lattice, DriftTEAPOT
from orbit.time_dep import time_dep
from orbit.time_dep.waveforms import SquareRootWaveform, ConstantWaveform
from orbit.utils import helper_funcs as hf

from helpers import get_traj, get_part_coords, track_part

sys.path.append('/Users/46h/Research/code/accphys/tools')
from utils import delete_files_not_folders
delete_files_not_folders('_output/')


# Initial settings
#------------------------------------------------------------------------------
use = {
    'collimator': True,
    'foil': True,
    'fringe': True,
    'kickers': True,
    'longitudinal impedence': True,
    'pyorbit diagnostics': True,
    'rf': True,
    'space charge': True,
    'solenoid': True,
}
x_foil = 0.0492 
y_foil = 0.0468
kin_energy = 0.800 # [GeV]
mass = 0.93827231 # [GeV/c^2]
turns = 1000
macros_per_turn = 260
intensity = 1.5e14

# Initial and final coordinates at s = 0
coords_s0_t0 = np.array([x_foil - 0.000, 0.000, y_foil - 0.000, 0.000])
coords_s0_t1 = np.array([x_foil - 25e-3, 0.000, y_foil - 0.000, 2.5e-3])

# Load SNS ring
if use['solenoid']:
    latfile = '_latfiles/SNSring_noRF_sol_nux6.18_nuy6.18.lat'
else:
    latfile = '_latfiles/SNSring_noRF_nux6.18_nuy6.18.lat'
latseq = 'rnginj'
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX(latfile, latseq)
ring.initialize()
ring.set_fringe(use['fringe'])
ring_length = ring.getLength()


print 'Switches:'
pprint(use)


# Beam setup
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
alpha_x = 0.063
alphay = 0.063
beta_x = 10.209 
betay = 10.776
emitlim = 0.152 * 2 * (order + 1) * 1e-6
xcenterpos = x_foil
ycenterpos = y_foil
xcentermom = ycentermom = 0.0
dist_x = JohoTransverse(order, alpha_x, beta_x, emitlim, xcenterpos, xcentermom)
dist_y = JohoTransverse(order, alphay, betay, emitlim, ycenterpos, ycentermom)

# Longitudinal linac distribution
zlim = (139.68 / 360.0) * ring_length
zmin, zmax = -zlim, zlim
tailfraction = 0.0
emean = sync_part.kinEnergy()
esigma = 0.0005
etrunc = 1.0
emin = sync_part.kinEnergy() - 0.0025
emax = sync_part.kinEnergy() + 0.0025
ecmean = 0.0
ecsigma = 0.000000001
ectrunc = 1.0
ecmin = -0.0035
ecmax = 0.0035
ecdrifti = 0.0
ecdriftf = 0.0
tturn = ring_length / (sync_part.beta() * 2.9979e8)
drifttime = 1000.0 * turns * tturn
ecparams = (ecmean, ecsigma, ectrunc, ecmin, ecmax, ecdrifti, ecdriftf, drifttime)
esnu = 100.0
esphase = 0.0
esmax = 0.0
nulltime = 0.0
esparams = (esnu, esphase, esmax, nulltime)
dist_z = SNSESpreadDist(ring_length, zmin, zmax, tailfraction, sync_part, emean, 
                        esigma, etrunc, emin, emax, ecparams, esparams)

## Uncomment for uniform longitudinal distribution
# eoffset = 0.0
# deltaEfrac = 0.0
# dist_z = UniformLongDist(zmin, zmax, sync_part, eoffset, deltaEfrac)


# Injection kickers
#------------------------------------------------------------------------------
kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']
param_names = ['kx', 'ky', 'kx', 'ky', 'ky', 'kx', 'ky', 'kx']
kicker_nodes = [ring.getNodeForName(name) for name in kicker_names]

# Maximum injection kicker angles at 1 GeV kinetic energy [mrad]
max_kicker_angles = 1.15 * np.array([12.84, 12.84, 7.13, 7.12, 7.12, 7.12, 12.84, 12.84])

# Scale angles based on actual kinetic energy
def get_momentum(kin_energy, mass):
    return np.sqrt(kin_energy * (kin_energy + 2*mass))
max_kicker_angles *= (get_momentum(1.0, mass) / get_momentum(kin_energy, mass))

# Convert from mrad to rad
max_kicker_angles *= 1e-3

# ARTIFICIALLY INCREASE KICKER LIMITS. The values seem to be less than the defaults,
# and with the provided kicker limits, we can't get to the closd orbit.
max_kicker_angles *= 4.0

def set_kicker_angles(angles, region='all'):
    """Set kicker angles in one half of the injection region."""
    if region == 1:
        lo, hi = 0, 4
    elif region == 2:
        lo, hi = 4, 8
    elif region == 'all':
        lo, hi = 0, 8
    for node, param_name, angle in zip(kicker_nodes[lo:hi], param_names[lo:hi], angles):
        node.setParam(param_name, angle)
        
def get_kicker_angles():
    return [node.getParam(param) for node, param in zip(kicker_nodes, param_names)]

def optimize_kickers(coords_s0, **kws):
    """Ensure closed orbit at s = 0 has [x, x', y, y'] = coords_s0.""" 
    kicker_angles = []
    for region in [2, 1]:
        if region == 2:
            sublattice = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
            bounds = (-max_kicker_angles[4:], max_kicker_angles[4:])
        elif region == 1:
            sublattice = hf.get_sublattice(ring, 'inj_start', None)
            sublattice.reverseOrder() # track backwards from s = 0 
            coords_s0[[1, 3]] *= -1 # initial slopes therefore change sign
            bounds = (-max_kicker_angles[:4], max_kicker_angles[:4])
            
        def penalty(angles):
            set_kicker_angles(angles, region)
            coords_outside_inj = track_part(sublattice, coords_s0, mass, kin_energy)
            return 1e6 * sum(np.square(coords_outside_inj))
        
        guess = np.zeros(4)
        result = opt.least_squares(penalty, guess, bounds=bounds, **kws)
        kicker_angles[:0] = result.x    
    return np.array(kicker_angles)

if use['kickers']:
    print 'Optimizing injection kickers.'
    kicker_angles_t0 = optimize_kickers(coords_s0_t0, verbose=1)  
    kicker_angles_t1 = optimize_kickers(coords_s0_t1, verbose=1)  
    ring.setLatticeOrder()
    t0 = 0.000 # [s]
    t1 = 0.001 # [s]
    amps_t0 = np.ones(8)
    amps_t1 = kicker_angles_t1 / kicker_angles_t0
    for node, amp_t0, amp_t1 in zip(kicker_nodes, amps_t0, amps_t1):
        waveform = SquareRootWaveform(sync_part, t0, t1, amp_t0, amp_t1)
        ring.setTimeDepNode(node.getParam('TPName'), waveform)
    set_kicker_angles(kicker_angles_t0)


# Black absorber collimator which acts as an aperture
#------------------------------------------------------------------------------
if use['collimator']:
    col_length = 0.00001
    ma = 9
    density_fac = 1.0
    shape = 1
    radius = 0.110
    pos = 0.5
    collimator = TeapotCollimatorNode(col_length, ma, density_fac, shape, 
                                      radius, 0., 0., 0., 0., pos, 'collimator1')
    addTeapotCollimatorNode(ring, 0.5, collimator)

    
# RF 
#------------------------------------------------------------------------------
if use['rf']:
    position1a = 183.0386827
    position1b = 185.3358827
    position1c = 187.6330827
    position2 = 189.9302827
    V1 = +0.0000133 # [MV]
    V2 = -0.0000200
    h1 = 1 # harmonic number (f/f0)
    h2 = 2
    phase1 = phase2 = 0.0
    ring.initialize()
    ztophi = 2 * np.pi / ring_length
    dEsync = 0.0
    length = 0.0
    rf1a_node = RFNode.Harmonic_RFNode(ztophi, dEsync, h1, V1, phase1, length, 'RF1')
    rf1b_node = RFNode.Harmonic_RFNode(ztophi, dEsync, h1, V1, phase1, length, 'RF1')
    rf1c_node = RFNode.Harmonic_RFNode(ztophi, dEsync, h1, V1, phase1, length, 'RF1')
    rf2_node = RFNode.Harmonic_RFNode(ztophi, dEsync, h2, V2, phase2, length, 'RF2')
    RFLatticeModifications.addRFNode(ring, position1a, rf1a_node)
    RFLatticeModifications.addRFNode(ring, position1b, rf1b_node)
    RFLatticeModifications.addRFNode(ring, position1c, rf1c_node)
    RFLatticeModifications.addRFNode(ring, position2,  rf2_node)


# Longitudinal impedence
#------------------------------------------------------------------------------
if use['longitudinal impedence']:
    length = ring_length
    min_n_macros = 1000
    n_bins= 128
    position = 124.0

    # SNS Longitudinal Impedance tables. EKicker impedance from private
    # communication with J.G. Wang. Seems to be for 7 of the 14 kickers
    # (not sure why). Impedance in Ohms/n. Kicker and RF impedances are
    # inductive with real part positive and imaginary is negative by Chao
    #definition.
    ZL_EKicker = [
        complex(42., -182),
        complex(35, -101.5),
        complex(30.3333, -74.6667),
        complex(31.5, -66.5),
        complex(32.2,-57.4),
        complex(31.5, -51.333),
        complex(31, -49),
        complex(31.5, -46.375),
        complex(31.8889, -43.556),
        complex(32.9, -40.6),
        complex(32.7273, -38.18),
        complex(32.25, -35.58),
        complex(34.46, -32.846),
        complex(35, -30.5),
        complex(35.4667, -28.),
        complex(36.75, -25.81),
        complex(36.647, -23.88),
        complex(36.944, -21.1667),
        complex(36.474, -20.263),
        complex(36.4, -18.55),
        complex(35.333, -17),
        complex(35, -14.95),
        complex(33.478, -13.69),
        complex(32.375, -11.67),
        complex(30.8, -10.08),
        complex(29.615, -8.077),
        complex(28.519, -6.74),
        complex(27.5, -5),
        complex(26.552, -4.103),
        complex(25.433, -3.266),
        complex(24.3871, -2.7),
        complex(23.40625, -2.18)
    ]
    ZL_RF = [
        complex(0.0, 0.0),
        complex(0.750, 0.0),
        complex(0.333,0.0),
        complex(0.250, 0.0),
        complex(0.200, 0.0),
        complex(0.167, 0.0),
        complex(3.214, 0.0),
        complex(0.188, 0.0),
        complex(0.167, 0.0),
        complex(0.150, 0.0),
        complex(1.000, 0.0),
        complex(0.125, 0.0),
        complex(0.115, 0.0),
        complex(0.143, 0.0),
        complex(0.333, 0.0),
        complex(0.313, 0.0),
        complex(0.294, 0.0),
        complex(0.278, 0.0),
        complex(0.263, 0.0),
        complex(0.250, 0.0),
        complex(0.714, 0.0),
        complex(0.682, 0.0),
        complex(0.652, 0.0),
        complex(0.625, 0.0),
        complex(0.600, 0.0),
        complex(0.577, 0.0),
        complex(0.536, 0.0),
        complex(0.536, 0.0),
        complex(0.517, 0.0),
        complex(0.500, 0.0),
        complex(0.484, 0.0),
        complex(0.469, 0.0)
    ]
    Z = []
    for zk, zrf in zip(ZL_EKicker, ZL_RF):
        zreal = zk.real / 1.75 + zrf.real
        zimag = zk.imag / 1.75 + zrf.imag
        Z.append(complex(zreal, zimag))

    impedancenode = LImpedance_Node(length, min_n_macros, n_bins)
    impedancenode.assignImpedance(Z)
    addImpedanceNode(ring, position, impedancenode)

    
# Space charge
#------------------------------------------------------------------------------
if use['space charge']:
    
    # Longitudinal
    b_a = 10.0 / 3.0
    length = ring_length
    min_n_macros = 1000
    n_long_slices = 128 
    position = 124.0
    zreal = (0.0)
    zimag = (0.0)
    Z = []
    for i in range(0,32):
        Z.append(complex(zreal, zimag))
    sc_node_long = SC1D_AccNode(b_a, length, min_n_macros, 1, n_long_slices)
    sc_node_long.assignImpedance(Z);
    addLongitudinalSpaceChargeNode(ring, position, sc_node_long)

    # Transverse
    ring.split(1.0) # at most 1 meter separation between calculations
    min_n_macros = 1000
    n_boundary_pts = 128
    n_free_space_modes = 32
    r_boundary = 0.220
    boundary = Boundary2D(n_boundary_pts, n_free_space_modes,
                          'Circle', r_boundary, r_boundary)
    sc_path_length_min = 0.00000001
    grid_size = (128, 128, 64) 
    sc_calc = SpaceChargeCalc2p5D(*grid_size)
    sc_nodes_trans = scLatticeModifications.setSC2p5DAccNodes(
        ring, sc_path_length_min, sc_calc, boundary)
        
        
# Injection
#------------------------------------------------------------------------------
thickness = 390.0
foil_xmin = xcenterpos - 0.0085
foil_xmax = xcenterpos + 0.0085
foil_ymin = ycenterpos - 0.0080
foil_ymax = ycenterpos + 0.100
foil_boundaries = [foil_xmin, foil_xmax, foil_ymin, foil_ymax]

injection_node = TeapotInjectionNode(macros_per_turn, bunch, lostbunch, 
                                     foil_boundaries, dist_x, dist_y, dist_z)
addTeapotInjectionNode(ring, 0.0, injection_node)

if use['foil']:
    foil_node = TeapotFoilNode(foil_xmin, foil_xmax, foil_ymin, foil_ymax, thickness)
    foil_node.setScatterChoice(2)
    addTeapotFoilNode(ring, 0.000001, foil_node)


# Diagnostics
#------------------------------------------------------------------------------
if use['pyorbit diagnostics']:
    tunes = TeapotTuneAnalysisNode("'tune_analysis'")
    tunes.assignTwiss(9.19025, -1.78574, -0.000143012, -2.26233e-05, \
                      8.66549, 0.538244)
    addTeapotDiagnosticsNode(ring, 51.1921, tunes)

    statlat = TeapotStatLatsNode('_output/data/statlats_pyorbit.dat')
    addTeapotDiagnosticsNode(ring, 0.2, statlat)

    order = 4
    moment = TeapotMomentsNode('_output/data/moments_pyorbit', order)
    addTeapotDiagnosticsNode(ring, 0.2, moment)

bunch_monitor_node = AnalysisNode(0.0, 'bunch_monitor', longitudinal=True)
injection_node.addChildNode(bunch_monitor_node, injection_node.EXIT)


# Run simulation
#------------------------------------------------------------------------------
print 'Tracking'
for _ in trange(turns):
    ring.trackBunch(bunch, params_dict)
    
coords = bunch_monitor_node.get_data('bunch_coords', 'all_turns')
for turn, X in enumerate(coords):
    np.save('_output/data/X_{}.npy'.format(turn), X)
    
    
# Save injection region closed orbit trajectory
#------------------------------------------------------------------------------
if use['kickers']:
    ring = hf.lattice_from_file(latfile, latseq)
    ring.split(0.01)
    inj_region1 = hf.get_sublattice(ring, 'inj_start', None)
    inj_region2 = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
    for i, kicker_angles in enumerate([kicker_angles_t0, kicker_angles_t1]):
        for node_name, param_name, angle in zip(kicker_names, param_names, kicker_angles):
            node = ring.getNodeForName(node_name)
            node.setParam(param_name, angle)
        coords1, positions1 = get_traj(inj_region1, [0, 0, 0, 0], mass, kin_energy)
        coords2, positions2 = get_traj(inj_region2, 1e-3 * coords1[-1], mass, kin_energy)
        coords = np.vstack([coords1, coords2])
        positions = np.hstack([positions1, positions2 + positions1[-1]])
        np.save('_output/data/inj_region_coords_t{}.npy'.format(i), coords)
        np.save('_output/data/inj_region_positions_t{}.npy'.format(i), positions)
        np.savetxt('_output/data/kicker_angles_t{}.dat'.format(i), kicker_angles)
