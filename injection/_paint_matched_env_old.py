"""This script simulates full injection in the SNS ring."""
from __future__ import print_function
import sys
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
from orbit.time_dep.waveforms import SquareRootWaveform, ConstantWaveform
from orbit.utils import helper_funcs as hf
from orbit.utils.general import save_stacked_array
from orbit.utils.general import delete_files_not_folders

from helpers import get_traj, get_part_coords, track_part



# Initial settings
#------------------------------------------------------------------------------
delete_files_not_folders('_output/')

use = {
    'collimator': False,
    'foil scattering': False,
    'fringe': False,
    'kickers': True,
    'longitudinal impedence': False,
    'pyorbit diagnostics': False,
    'rf': False,
    'skew quads': False,
    'solenoid': False,
    'space charge': False,
    'uniform longitudinal distribution': True,
}
x_foil = 0.0492 
y_foil = 0.0468
kin_energy = 1.00 # [GeV]
mass = 0.93827231 # [GeV/c^2]
turns = 1000
macros_per_turn = 260
intensity = 1.5e14

# Initial and final coordinates at injection point
inj_coords_t0 = np.array([x_foil - 10e-3, 0.0, y_foil - 9e-3, 0.0])
inj_coords_t1 = np.array([x_foil - 25e-3, 0.0, y_foil - 30e-3, 0.0])

print('Switches:')
pprint(use)


# Lattice setup
#------------------------------------------------------------------------------
# Load SNS ring
if use['solenoid']:
    madx_file = '_input/SNSring_noRF_sol_nux6.18_nuy6.18.lat'
else:
    madx_file = '_input/SNSring_noRF_nux6.18_nuy6.18.lat'
madx_file = '_input/SNSring_noRF_nux6.18_nuy6.18.lat'
madx_seq = 'rnginj'
ring = time_dep.TIME_DEP_Lattice()
ring.readMADX(madx_file, madx_seq)
ring.set_fringe(False)
ring.initialize()
ring_length = ring.getLength()

# Toggle skew quads
def get_skew_quad_nodes(ring):
    skew_quad_nodes = []
    for node in ring.getNodes():
        name = node.getName()
        if name.startswith('qsc'):
            node.setParam('skews', [0, 1])
            skew_quad_nodes.append(node)
    return skew_quad_nodes
        
def set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths):
    for node, strength in zip(skew_quad_nodes, skew_quad_strengths):
        node.setParam('kls', [0.0, strength])
        
if use['skew quads']:
    skew_quad_nodes = get_skew_quad_nodes(ring)
    env_skew_quad_nodes = get_skew_quad_nodes(env_ring)
    skew_quad_strengths = np.zeros(len(skew_quad_nodes))
    skew_quad_strengths[12] = 0.1
    set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths)
    set_skew_quad_strengths(env_skew_quad_nodes, skew_quad_strengths)

    
# Envelope matching
#------------------------------------------------------------------------------
print('Matching envelope.')

eps_l = 50e-6 # nonzero intrinsic emittance [m rad]
mode = 1
eps_x_frac = 0.6
bunch_length = (139.68 / 360.0) * ring_length
env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, bunch_length)
if use['space charge']:
    env.set_intensity(intensity)
else:
    env.set_intensity(0.0)
env_ring = TEAPOT_Lattice() # for envelope tracking
env_ring.readMADX(madx_file, madx_seq)
env_ring.set_fringe(False)
max_solver_spacing = 1.0
env_solver_nodes = set_env_solver_nodes(env_ring, env.perveance, max_solver_spacing)
env.match(env_ring, env_solver_nodes, method='lsq', verbose=2)
env.print_twiss4D()

alpha_lx, alpha_ly, beta_lx, beta_ly, u, nu = env.twiss4D()

if mode == 1:
    v_l = np.array([
        np.sqrt(beta_lx),
        -(alpha_lx + 1j*(1 - u)) / np.sqrt(beta_lx),
        np.sqrt(beta_ly) * np.exp(1j * nu),
        -((alpha_ly + 1j*u) / np.sqrt(beta_ly)) * np.exp(1j * nu),
    ])
elif mode == 2:
    v_l = np.array([
        np.sqrt(beta_lx) * np.exp(1j * nu),
        -((alpha_lx + 1j*u) / np.sqrt(beta_lx)) * np.exp(1j * nu),
        np.sqrt(beta_ly),
        -(alpha_ly + 1j*(1 - u)) / np.sqrt(beta_ly)
    ])
    
phase = np.radians(0.0)
final_coords = np.real(np.sqrt(4 * eps_l) * v_l * np.exp(-1j * phase))
    
# Find phase which minimizes required slope from kickers
smallest_max_abs_slope = np.inf
eigvec_phase = 0.0
final_coords = np.zeros(4)
for phase in np.linspace(0.0, 2 * np.pi, 100):
    x, xp, y, yp = np.real(np.sqrt(4 * eps_l) * v_l * np.exp(-1j * phase))
    max_abs_slope = max(abs(xp), abs(yp))
    if max_abs_slope < smallest_max_abs_slope:
        smallest_max_abs_slope = max_abs_slope
        eigvec_phase = phase
        final_coords = np.array([x, xp, y, yp])

inj_coords_t1 = np.subtract([x_foil, 0.0, y_foil, 0.0], final_coords)
print('final coords (foil frame) [mm mrad]:', 1e3 * final_coords)
print('inj_coords_t0:', 1e3 * inj_coords_t0)
print('inj_coords_t1:', 1e3 * inj_coords_t1)
np.save('matched_eigenvector', v_l)
np.save('matched_env_params', env.params)

