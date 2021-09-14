"""This script simulates full injection in the SNS ring."""
from __future__ import print_function
import sys
import math
from pprint import pprint

import numpy as np
import scipy.optimize as opt
from tqdm import tqdm, trange

from bunch import Bunch
from foil import Foil
from impedances import LImpedance, TImpedance
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D
from orbit.collimation import TeapotCollimatorNode, addTeapotCollimatorNode
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import BunchStatsNode
from orbit.diagnostics import analysis
from orbit.diagnostics import add_analysis_node
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import StatLats, Moments
from orbit.diagnostics import TeapotStatLatsNode
from orbit.diagnostics import TeapotMomentsNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.diagnostics import addTeapotStatLatsNodeSet
from orbit.diagnostics import addTeapotMomentsNodeSet
from orbit.envelope import DanilovEnvelope
from orbit.foils import TeapotFoilNode, addTeapotFoilNode
from orbit.impedances import LImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import addImpedanceNode
from orbit.injection import TeapotInjectionNode
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import JohoTransverse, JohoLongitudinal
from orbit.injection import SNSESpreadDist, UniformLongDist
from orbit.rf_cavities import RFNode, RFLatticeModifications
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from orbit.teapot import teapot, TEAPOT_Lattice, DriftTEAPOT
from orbit.time_dep import time_dep
from orbit.time_dep.waveforms import SquareRootWaveform
from orbit.time_dep.waveforms import ConstantWaveform
from orbit.time_dep.waveforms import LinearWaveform
from orbit.utils import helper_funcs as hf
from orbit.utils.consts import speed_of_light
from orbit.utils.general import save_stacked_array
from orbit.utils.general import delete_files_not_folders

# Local
from helpers import get_traj
from helpers import get_part_coords
from helpers import InjRegionController


# Switches
#------------------------------------------------------------------------------
switches = {
    'orbit corrector bump': True,
    'equal emittances': True,
    'fringe': True,
    'solenoid': True,
    'space charge': True,
    'impedence': True,
    'foil scattering': True,
    'rf': True,
    'collimator': True,
    'no energy spread': False,
}

print('Switches:')
pprint(switches)


# Initial settings
#------------------------------------------------------------------------------
print("Removing data in '_output/data/' folder.")
delete_files_not_folders('_output/data/')

madx_file = '_input/SNSring_nux6.18_nuy6.18.lat'
madx_seq = 'rnginj'
X_FOIL = 0.0486 # [m]
Y_FOIL = 0.0460 # [m]
kin_energy = 0.8 # [GeV]
mass = 0.93827231 # [GeV/c^2]
n_stored_turns = 0
n_inj_turns = 300
intensity = 1.5e14 * float(n_inj_turns) / 1000.
# intensity = 0.35e14
macros_per_turn = 2000
macro_size = intensity / n_inj_turns / macros_per_turn

# Initial and final coordinates at injection point
inj_coords_t0 = np.array([
    X_FOIL,
    0.0,
    Y_FOIL,
    0.0,
])
inj_coords_t1 = np.array([
    X_FOIL - 0.020,
    -0.000,
    Y_FOIL - 0.000,
    -0.001744,
])


# Lattice setup
#------------------------------------------------------------------------------
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX(madx_file, madx_seq)
ring.set_fringe(False)
ring.initialize()
ring_length = ring.getLength()

if switches['equal emittances']:
    alpha_x, alpha_y, beta_x, beta_y = hf.twiss_at_injection(ring, mass, kin_energy)
    print(alpha_x, alpha_y, beta_x, beta_y)
    dx_dyp_ratio = np.sqrt((beta_x * beta_y) / (1 + alpha_y**2))
    abs_dyp = abs(inj_coords_t1[3])
    abs_dx = dx_dyp_ratio * abs_dyp
    inj_coords_t1[0] = X_FOIL - abs_dx
    print('Setting dx/dyp for equal emittances.')
    print('  |dyp| = {}'.format(abs_dyp))
    print('  dx/dyp = {}'.format(dx_dyp_ratio))
    print('  |dx| = {}'.format(abs_dx))
    
    
# For some reason, calling `node.setParam('B', 0.)` on the solenoid node gives 
# NAN for the orbit. I don't know the issue. The solution for now is to load 
# an identical lattice without the solenoid.
if switches['solenoid']:
    madx_file = '_input/SNSring_nux6.18_nuy6.18_solenoid.lat'
print("MADX file = '{}'".format(madx_file))
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX(madx_file, madx_seq)
ring.set_fringe(False)
ring.initialize()
ring_length = ring.getLength()
print('ring length = {}'.format(ring_length))

    
# Beam setup
#------------------------------------------------------------------------------
# Initialize bunch
bunch = Bunch()
bunch.mass(mass)
bunch.macroSize(macro_size)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(kin_energy)
lostbunch = Bunch()
lostbunch.addPartAttr('LostParticleAttributes')
params_dict = {'bunch':bunch, 'lostbunch':lostbunch}

# Transverse linac distribution
inj_center_x = X_FOIL
inj_center_y = Y_FOIL
inj_center_xp = 0.0
inj_center_yp = 0.0
alpha_x = 0.049 # [rad]
alpha_y = 0.058 # [rad]
beta_x = 13.072 # [m/rad]
beta_y = 11.996 # [m/rad]
order_x = 9.
order_y = 9.
eps_x_rms = 0.221e-6 # [m rad]
eps_y_rms = 0.221e-6 # [m rad]
eps_x_lim = eps_x_rms * 2. * (order_x + 1.)
eps_y_lim = eps_y_rms * 2. * (order_y + 1.)
dist_x = JohoTransverse(order_x, alpha_x, beta_x, eps_x_lim, inj_center_x, inj_center_xp)
dist_y = JohoTransverse(order_y, alpha_y, beta_y, eps_y_lim, inj_center_y, inj_center_yp)

# Longitudinal linac distribution
zlim = (135.0 / 360.0) * ring_length
zmin, zmax = -zlim, zlim
tailfraction = 0.0
emean = sync_part.kinEnergy()
efac = 0.784
esigma = 0.0015 * efac
etrunc = 1.0
emin = sync_part.kinEnergy() - 0.0025 * efac
emax = sync_part.kinEnergy() + 0.0025 * efac
ecmean = 0.0
ecsigma = 0.0015 * efac
ectrunc = 1.0
ecmin = -0.0035 * efac
ecmax = 0.0035 * efac
ecdrifti = 0.0
ecdriftf = 0.0

# From Holmes 2018
#------------------------------------
esigma = 0.0005
emin = sync_part.kinEnergy() - 0.0025
emax = sync_part.kinEnergy() + 0.0025
ecmean = 0.0
ecsigma = 0.000000000001
ecmin = -0.0035
ecmax = +0.0035
#------------------------------------

seconds_per_turn = ring_length / (sync_part.beta() * speed_of_light)
drifttime = 1000. * n_inj_turns * seconds_per_turn # [ms]
ecparams = (ecmean, ecsigma, ectrunc, ecmin, ecmax, 
            ecdrifti, ecdriftf, drifttime)
esnu = 100.0
esphase = 0.0
esmax = 0.0
nulltime = 0.0
esparams = (esnu, esphase, esmax, nulltime)
dist_z = SNSESpreadDist(ring_length, zmin, zmax, tailfraction, sync_part, 
                        emean, esigma, etrunc, emin, emax, 
                        ecparams, esparams)

if switches['no energy spread']:
    eoffset = 0.
    deltaEfrac = 0.
    dist_z = UniformLongDist(-0.5 * ring_length, 0.5 * ring_length, 
                             sync_part, eoffset, deltaEfrac)


# Injection kickers
#-------------------------------------------------------------------------------
t0 = 0.000 # injection start time [s]
t1 = n_inj_turns * seconds_per_turn # injection stop time [s]

inj_controller = InjRegionController(ring, mass, kin_energy)

if switches['orbit corrector bump']:
    inj_controller.bump_vertical_orbit(max_nfev=5000, verbose=2)
    
corrector_angles = inj_controller.get_corrector_angles()
solver_kws = dict(max_nfev=1000, verbose=2)
kicker_angles_t0 = inj_controller.set_coords_at_foil(inj_coords_t0, **solver_kws)
kicker_angles_t1 = inj_controller.set_coords_at_foil(inj_coords_t1, **solver_kws)
inj_controller.set_kicker_angles(kicker_angles_t0)

ring.setLatticeOrder()
amps_t0 = np.ones(8)
amps_t1 = np.abs(kicker_angles_t1 / kicker_angles_t0)
for node, amp_t0, amp_t1 in zip(inj_controller.kicker_nodes, amps_t0, amps_t1):
    waveform = SquareRootWaveform(sync_part, t0, t1, amp_t0, amp_t1)
    ring.setTimeDepNode(node.getParam('TPName'), waveform)
    
    
# # Injection node and foil node
# #------------------------------------------------------------------------------
# thickness = 400.0
# foil_xmin = X_FOIL - 0.0085
# foil_xmax = X_FOIL + 0.0085
# foil_ymin = Y_FOIL - 0.0080
# foil_ymax = Y_FOIL + 0.100
# foil_boundaries = [foil_xmin, foil_xmax, foil_ymin, foil_ymax]

# injection_node = TeapotInjectionNode(
#     macros_per_turn, bunch, lostbunch, foil_boundaries, 
#     dist_x, dist_y, dist_z, 
#     nmaxmacroparticles=macros_per_turn*n_inj_turns
# )
# addTeapotInjectionNode(ring, 0.0, injection_node)

# if switches['foil scattering']:
#     foil_node = TeapotFoilNode(foil_xmin, foil_xmax, foil_ymin, foil_ymax, thickness)
#     foil_node.setScatterChoice(2)
#     addTeapotFoilNode(ring, 0.000001, foil_node)


# # Black absorber collimator to act as an aperture
# #------------------------------------------------------------------------------
# if switches['collimator']:
#     col_length = 0.00001
#     ma = 9
#     density_fac = 1.0
#     shape = 1
#     radius = 0.110
#     pos = 0.5
#     collimator = TeapotCollimatorNode(col_length, ma, density_fac, shape, 
#                                       radius, 0., 0., 0., 0., pos, 'collimator1')
#     addTeapotCollimatorNode(ring, 0.5, collimator)

    
# # RF cavities
# #------------------------------------------------------------------------------
# if switches['rf']:
#     ZtoPhi = 2.0 * math.pi / ring_length;
#     dESync = 0.0
#     RF1HNum = 1.0
#     RF1Voltage = +0.000004 # [GV]
#     RF1Phase = 0.0
#     RF2HNum = 2.0
#     RF2Voltage = -0.000004 # [GV]
#     RF2Phase = 0.0
#     length = 0.0
#     rf1_node = RFNode.Harmonic_RFNode(ZtoPhi, dESync, RF1HNum, RF1Voltage, 
#                                       RF1Phase, length, "RF1")
#     rf2_node = RFNode.Harmonic_RFNode(ZtoPhi, dESync, RF2HNum, RF2Voltage, 
#                                       RF2Phase, length, "RF2")
#     position1 = 188.7
#     position2 = 191.9
#     RFLatticeModifications.addRFNode(ring, position1, rf1_node)
#     RFLatticeModifications.addRFNode(ring, position2, rf2_node)

    
# # Longitudinal impedence
# #------------------------------------------------------------------------------
# if switches['impedence']:
#     length = ring_length
#     min_n_macros = 1000
#     n_bins = 128
#     position = 124.0

#     # SNS Longitudinal Impedance tables. EKicker impedance from private
#     # communication with J.G. Wang. Seems to be for 7 of the 14 kickers
#     # (not sure why). Impedance in Ohms/n. Kicker and RF impedances are
#     # inductive with real part positive and imaginary is negative by Chao
#     # definition.
#     ZL_EKicker = [
#         complex(42., -182),
#         complex(35, -101.5),
#         complex(30.3333, -74.6667),
#         complex(31.5, -66.5),
#         complex(32.2,-57.4),
#         complex(31.5, -51.333),
#         complex(31, -49),
#         complex(31.5, -46.375),
#         complex(31.8889, -43.556),
#         complex(32.9, -40.6),
#         complex(32.7273, -38.18),
#         complex(32.25, -35.58),
#         complex(34.46, -32.846),
#         complex(35, -30.5),
#         complex(35.4667, -28.),
#         complex(36.75, -25.81),
#         complex(36.647, -23.88),
#         complex(36.944, -21.1667),
#         complex(36.474, -20.263),
#         complex(36.4, -18.55),
#         complex(35.333, -17),
#         complex(35, -14.95),
#         complex(33.478, -13.69),
#         complex(32.375, -11.67),
#         complex(30.8, -10.08),
#         complex(29.615, -8.077),
#         complex(28.519, -6.74),
#         complex(27.5, -5),
#         complex(26.552, -4.103),
#         complex(25.433, -3.266),
#         complex(24.3871, -2.7),
#         complex(23.40625, -2.18)
#     ]
#     ZL_RF = [
#         complex(0.0, 0.0),
#         complex(0.750, 0.0),
#         complex(0.333,0.0),
#         complex(0.250, 0.0),
#         complex(0.200, 0.0),
#         complex(0.167, 0.0),
#         complex(3.214, 0.0),
#         complex(0.188, 0.0),
#         complex(0.167, 0.0),
#         complex(0.150, 0.0),
#         complex(1.000, 0.0),
#         complex(0.125, 0.0),
#         complex(0.115, 0.0),
#         complex(0.143, 0.0),
#         complex(0.333, 0.0),
#         complex(0.313, 0.0),
#         complex(0.294, 0.0),
#         complex(0.278, 0.0),
#         complex(0.263, 0.0),
#         complex(0.250, 0.0),
#         complex(0.714, 0.0),
#         complex(0.682, 0.0),
#         complex(0.652, 0.0),
#         complex(0.625, 0.0),
#         complex(0.600, 0.0),
#         complex(0.577, 0.0),
#         complex(0.536, 0.0),
#         complex(0.536, 0.0),
#         complex(0.517, 0.0),
#         complex(0.500, 0.0),
#         complex(0.484, 0.0),
#         complex(0.469, 0.0)
#     ]
#     Z = []
#     for zk, zrf in zip(ZL_EKicker, ZL_RF):
#         zreal = zk.real / 1.75 + zrf.real
#         zimag = zk.imag / 1.75 + zrf.imag
#         Z.append(complex(zreal, zimag))

#     impedancenode = LImpedance_Node(length, min_n_macros, n_bins)
#     impedancenode.assignImpedance(Z)
#     addImpedanceNode(ring, position, impedancenode)

    
# # Space charge
# #------------------------------------------------------------------------------
# if switches['space charge']:
    
#     # Longitudinal
#     b_a = 10.0 / 3.0
#     length = ring_length
#     use_spacecharge = 1
#     min_n_macros = 1000
#     n_long_slices = 128 
#     position = 124.0
#     zreal = (0.0)
#     zimag = (0.0)
#     Z = []
#     for i in range(32):
#         Z.append(complex(zreal, zimag))
#     sc_node_long = SC1D_AccNode(b_a, length, min_n_macros, 
#                                 use_spacecharge, n_long_slices)
#     sc_node_long.assignImpedance(Z)
#     addLongitudinalSpaceChargeNode(ring, position, sc_node_long)

#     # Transverse
#     ring.split(1.0) # at most 1 meter separation between calculations
#     n_boundary_pts = 128
#     n_free_space_modes = 32
#     r_boundary = 0.22
#     boundary = Boundary2D(n_boundary_pts, n_free_space_modes, 
#                           'Circle', r_boundary, r_boundary)
#     grid_size = (128, 128, 128) 
#     sc_path_length_min = 0.00000001
#     sc_calc = SpaceChargeCalc2p5D(*grid_size)
#     sc_nodes_trans = scLatticeModifications.setSC2p5DAccNodes(
#         ring, sc_path_length_min, sc_calc, boundary)


# # Diagnostics
# #------------------------------------------------------------------------------
# bunch_monitor_node = BunchMonitorNode(mm_mrad=True, transverse_only=False)
# injection_node.addChildNode(bunch_monitor_node, injection_node.EXIT)


# # Run simulation
# #------------------------------------------------------------------------------
# ring.set_fringe(switches['fringe'])

# print('Painting...')
# for _ in trange(n_inj_turns):
#     ring.trackBunch(bunch, params_dict)
    
# print('Stored turns...')
# for _ in trange(n_stored_turns):
#     ring.trackBunch(bunch, params_dict)
    
# print('Saving turn-by-turn coordinates...')
# coords = bunch_monitor_node.get_data()
# save_stacked_array('_output/data/coords.npz', coords)

# print('Saving final lost bunch.')
# lostbunch.dumpBunch('_output/data/lostbunch.dat')
    
    
# Save injection region closed orbit trajectory (this should just be a method
# in the InjectionController class.)
#------------------------------------------------------------------------------
ring = TEAPOT_Lattice()
ring.readMADX(madx_file, madx_seq)
ring.set_fringe(False)
ring.split(0.01)
inj_controller = InjRegionController(ring, mass, kin_energy)
inj_controller.set_corrector_angles(corrector_angles)
inj_region1 = hf.get_sublattice(ring, 'inj_start', None)
inj_region2 = hf.get_sublattice(ring, 'inj_mid', 'inj_end')
for i, kicker_angles in enumerate([kicker_angles_t0, kicker_angles_t1]):
    inj_controller.set_kicker_angles(kicker_angles)
    coords1, positions1 = get_traj(inj_region1, [0, 0, 0, 0], mass, kin_energy)
    coords2, positions2 = get_traj(inj_region2, coords1[-1], mass, kin_energy)
    coords = np.vstack([coords1, coords2])
    positions = np.hstack([positions1, positions2 + positions1[-1]])
    np.save('_output/data/inj_region_coords_t{}.npy'.format(i), coords)
    np.save('_output/data/inj_region_positions_t{}.npy'.format(i), positions)
    np.savetxt('_output/data/kicker_angles_t{}.dat'.format(i), kicker_angles)
    
# Save simulation info
file = open('_output/data/info.txt', 'w')
for key in sorted(list(switches)):
    file.write('{} = {}\n'.format(key, switches[key]))
file.write('madx_file = {}\n'.format(madx_file))
file.write('madx_seq = {}\n'.format(madx_seq))
file.write('kin_energy = {} [GeV]\n'.format(kin_energy))
file.write('mass = {} [GeV/c^2]\n'.format(mass))
file.write('intensity = {}\n'.format(intensity))
file.write('n_inj_turns = {}\n'.format(n_inj_turns))
file.write('n_stored_turns = {}\n'.format(n_stored_turns))
file.write('macros_per_turn = {}\n'.format(macros_per_turn))
file.write('t0 = {}\n'.format(t0))
file.write('t1 = {}\n'.format(t1))
file.write('inj_coords_t0 = ({}, {}, {}, {})\n'.format(*inj_coords_t0))
file.write('inj_coords_t1 = ({}, {}, {}, {})\n'.format(*inj_coords_t1))
file.write('X_FOIL = {}\n'.format(X_FOIL))
file.write('Y_FOIL = {}\n'.format(X_FOIL))
file.close()