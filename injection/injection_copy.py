"""Simulation of injection in the SNS ring.

NOTES
--------------------------------------------------------------------------------
1. Do not use transverse impedance calculation unless a sliced space charge 
   model is used. 
2. If fringe fields are to be turned on in the simulation, then they must be
   turned on when optimizing the injection region magnets to obtain certain
3. Need to be careful about space charge model when RF voltages are changed. 
   If longitudinal profile has two peaks, the horizontal distribution becomes
   hollow at certain points along the beam line when using the 2.5D solver.
""" 
from __future__ import print_function
import sys
from pprint import pprint

import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
from tqdm import trange

from bunch import Bunch
from foil import Foil
from impedances import LImpedance
from impedances import TImpedance
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D
from orbit.collimation import TeapotCollimatorNode
from orbit.collimation import addTeapotCollimatorNode
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
from orbit.foils import TeapotFoilNode
from orbit.foils import addTeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import LImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.impedances import FreqDep_TImpedance_Node
from orbit.impedances import BetFreqDep_TImpedance_Node
from orbit.injection import TeapotInjectionNode
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import JohoTransverse
from orbit.injection import JohoLongitudinal
from orbit.injection import SNSESpreadDist
from orbit.injection import UniformLongDist
from orbit.lattice import AccNode
from orbit.rf_cavities import RFNode, RFLatticeModifications
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge import sc2p5d
from orbit.space_charge import sc2dslicebyslice
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
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
    'equal emittances': False,
    'solenoid': False,
    'fringe': True,
    'transverse space charge': 'sliced', # {'2.5D', 'sliced', False}
    'longitudinal space charge': True,
    'transverse impedance': True,
    'longitudinal impedance': True,
    'foil scattering': True,
    'rf': True,
    'collimator': True,
}

print('Switches:')
pprint(switches)


madx_file = '_input/SNSring_nux6.18_nuy6.18_foilinbend.lat'
madx_seq = 'rnginj'
X_FOIL = 0.0486 # [m]
Y_FOIL = 0.0460 # [m]
kin_energy = 0.8 # [GeV]
mass = 0.93827231 # [GeV/c^2]
bunch_length_frac = (30.0 / 64.0) 


    
eps_l = 20e-6 # nonzero intrinsic emittance [m rad]

for n_inj_turns in [300, 400, 500]:
    
    
    default_minipulse_intensity = 1.5e14 / 1000.
    default_bunch_length_frac = (50.0 / 64.0)
    minipulse_intensity = default_minipulse_intensity * (bunch_length_frac / default_bunch_length_frac)
    intensity = minipulse_intensity * n_inj_turns
    
    
    ring = TEAPOT_Lattice() # for envelope tracking
    ring.readMADX(madx_file, madx_seq)
    ring.set_fringe(False)
    ring_length = ring.getLength()

    mode = 1
    eps_x_frac = 0.5
    ring_length = ring.getLength()
    bunch_length = bunch_length_frac * ring_length
    bunch_length = (bunch_length / ring_length) * ring_length
    env = DanilovEnvelope(eps_l, mode, eps_x_frac, mass, kin_energy, bunch_length)
    env.set_intensity(intensity)
    
    bunch_length = bunch_length_frac * ring_length
    max_solver_spacing = 1.0
    env_solver_nodes = set_env_solver_nodes(ring, env.perveance, max_solver_spacing)
    
    env.match(ring, env_solver_nodes, method='lsq', verbose=2)
    print('n_inj_turns = {}'.format(n_inj_turns))
    env.print_twiss2D()
    env.print_twiss4D()
    print()