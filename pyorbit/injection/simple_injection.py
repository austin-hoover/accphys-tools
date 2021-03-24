"""
SNS injection simulation.
"""

import math
import sys
import numpy as np
from tqdm import tqdm, trange

from bunch import Bunch
from orbit.utils.orbit_mpi_utils import (
    bunch_orbit_to_pyorbit, bunch_pyorbit_to_orbit)

from orbit.analysis import AnalysisNode
from orbit.utils import helper_funcs as hf

from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot, TEAPOT_Lattice, DriftTEAPOT

from orbit.time_dep import time_dep
from orbit.time_dep.waveform import ConstantWaveform, SquareRootWaveform
from orbit.kickernodes import (
    XKicker, YKicker, 
    TeapotXKickerNode, TeapotYKickerNode, addTeapotKickerNode,
    rootTWaveform, flatTopWaveform)

from orbit.injection import (
    TeapotInjectionNode, addTeapotInjectionNode, InjectParts,
    JohoTransverse, JohoLongitudinal, SNSESpreadDist)

from foil import Foil
from orbit.foils import TeapotFoilNode, addTeapotFoilNode

from orbit.collimation import TeapotCollimatorNode, addTeapotCollimatorNode

from orbit.rf_cavities import RFNode, RFLatticeModifications

from spacecharge import Boundary2D
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from spacecharge import SpaceChargeCalc2p5D, Boundary2D
from spacecharge import LSpaceChargeCalc
from orbit.space_charge.sc1d import (
    addLongitudinalSpaceChargeNode, SC1D_AccNode)

from orbit.diagnostics import (
    StatLats, Moments, addTeapotDiagnosticsNode,
    TeapotStatLatsNode, TeapotMomentsNode, TeapotTuneAnalysisNode,
    addTeapotStatLatsNodeSet, addTeapotMomentsNodeSet,
    profiles)

sys.path.append('/Users/46h/Research/code/accphys/tools')
from utils import delete_files_not_folders
delete_files_not_folders('_output/')


# Set up bunch
#------------------------------------------------------------------------------
intensity = 7.8e13
turns = 1000.0
NTURNS = 1000
NDUMPS = 5
macros_per_turn = 260
macrosize = intensity / turns / macros_per_turn

bunch = Bunch()
bunch.mass(0.93827231)
bunch.macroSize(macrosize)
energy = 1.0 #Gev
bunch.getSyncParticle().kinEnergy(energy)

lostbunch = Bunch()
lostbunch.addPartAttr('LostParticleAttributes')

params_dict = {'bunch':bunch, 'lostbunch':lostbunch}


# Load SNS accumulator ring
#------------------------------------------------------------------------------
lattice = time_dep.TIME_DEP_Lattice()

lattice.readMAD('_latfiles/SNSring_pyOrbitBenchmark.LAT', 'RING')
lattice.setLatticeOrder()
lattice.initialize()

latt_name = lattice.getName()
latt_length = lattice.getLength()
n_nodes = len(lattice.getNodes())


# Make linac x and y distribution functions
#------------------------------------------------------------------------------
sp = bunch.getSyncParticle()

order = 3.0
alphax = 0.063
betax = 10.209
alphay = 0.063
betay = 10.776
emitlim = 0.152 * 2 * (order + 1) * 1e-6
xcenterpos = 0.0468
xcentermom = 0.0
ycenterpos = 0.0492
ycentermom = 0.0

zlim = 120.0 * latt_length / 360.0
zmin = -zlim
zmax = zlim
tailfraction = 0.0
emean = sp.kinEnergy()
efac = 0.784
esigma = 0.0015 * efac
etrunc = 1.0
emin = sp.kinEnergy() - 0.0025 * efac
emax = sp.kinEnergy() + 0.0025 * efac
ecmean = 0.0
ecsigma = 0.0015 * efac
ectrunc = 1.0
ecmin = -0.0035 * efac
ecmax = 0.0035 * efac
ecdrifti = 0.0
ecdriftf = 0.0
turns = 1000.0
tturn = latt_length / (sp.beta() * 2.9979e8)
drifttime= 1000.0 * turns * tturn
ecparams = (ecmean, ecsigma, ectrunc, ecmin, ecmax, ecdrifti, 
            ecdriftf, drifttime)
esnu = 100.0
esphase = 0.0
esmax = 0.0
nulltime = 0.0
esparams = (esnu, esphase, esmax, nulltime)

xFunc = JohoTransverse(order, alphax, betax, emitlim, xcenterpos, xcentermom)
yFunc = JohoTransverse(order, alphay, betay, emitlim, ycenterpos, ycentermom)
lFunc = SNSESpreadDist(latt_length, zmin, zmax, tailfraction, sp, emean, 
                       esigma, etrunc, emin, emax, ecparams, esparams)


# Add injection kickers
#------------------------------------------------------------------------------
# k_hkicker10 = 7.211536E-03
# k_vkicker10 = 4.188402E-03
# k_hkicker11 = -2.278306E-03
# k_vkicker11 = -2.118213E-03
# k_hkicker12 = k_hkicker11
# k_vkicker12 = k_vkicker11
# k_hkicker13 = k_hkicker10
# k_vkicker13 = k_vkicker10

# hkick10 = lattice.getNodeForName('IKICKH_A10')
# vkick10 = lattice.getNodeForName('IKICKV_A10')
# hkick11	= lattice.getNodeForName('IKICKH_A11')
# vkick11 = lattice.getNodeForName('IKICKV_A11')
# hkick12 = lattice.getNodeForName('IKICKH_A12')
# vkick12 = lattice.getNodeForName('IKICKV_A12')
# hkick13	= lattice.getNodeForName('IKICKH_A13')
# vkick13 = lattice.getNodeForName('IKICKV_A13')

# hkick10.setParam('kx', k_hkicker10)
# vkick10.setParam('ky', k_vkicker10)
# hkick11.setParam('kx', k_hkicker11)
# vkick11.setParam('ky', k_vkicker11)
# hkick12.setParam('kx', k_hkicker12)
# vkick12.setParam('ky', k_vkicker12)
# hkick13.setParam('kx', k_hkicker13)
# vkick13.setParam('ky', k_vkicker13)

# sp = bunch.getSyncParticle()
# t1 = 0.0 # [s]
# t2 = 0.001 # [s]
# t1amp = 1.0
# t2amp = 0.58

# xkickerwave = SquareRootWaveform(sp, latt_length, t1, t2, t1amp, t2amp)
# ykickerwave = SquareRootWaveform(sp, latt_length, t1, t2, t1amp, t2amp)

# lattice.setTimeDepNode('IKICKH_A10_1', xkickerwave)
# lattice.setTimeDepNode('IKICKV_A10_1', ykickerwave)
# lattice.setTimeDepNode('IKICKH_A11_1', xkickerwave)
# lattice.setTimeDepNode('IKICKV_A11_1', ykickerwave)
# lattice.setTimeDepNode('IKICKH_A12_1', xkickerwave)
# lattice.setTimeDepNode('IKICKV_A12_1', ykickerwave)
# lattice.setTimeDepNode('IKICKH_A13_1', xkickerwave)
# lattice.setTimeDepNode('IKICKV_A13_1', ykickerwave)

kicker_names = ['IKICKH_A10', 'IKICKH_A11', 'IKICKH_A12', 'IKICKH_A13', 
                'IKICKV_A10', 'IKICKV_A11', 'IKICKV_A12', 'IKICKV_A13']
kicker_coeff = {
    'IKICKH_A10': 0.844547,
    'IKICKH_A11': -0.44845,
    'IKICKH_A12': -0.47060,
    'IKICKH_A13': 0.912197,
    'IKICKV_A10': 0.840809,
    'IKICKV_A11': -0.46629,
    'IKICKV_A12': -0.44012,
    'IKICKV_A13': 0.817781
}
kicker_voltages = {
    't0': {
        'IKICKH_A10': 18.44372,
        'IKICKH_A11': 11.89463,
        'IKICKH_A12': 14.34026,
        'IKICKH_A13': 19.95128,
        'IKICKV_A10': 11.23115,
        'IKICKV_A11': 9.709696,
        'IKICKV_A12': 19.70032,
        'IKICKV_A13': 15.64662},
    't1': {
        'IKICKH_A10': 12.24726,
        'IKICKH_A11': 7.904081,
        'IKICKH_A12': 10.48900,
        'IKICKH_A13': 14.18529,
        'IKICKV_A10': 15.46127,
        'IKICKV_A11': 19.83746,
        'IKICKV_A12': 8.153721,
        'IKICKV_A13': 10.45873}
}
kicker_angles = {
    't0': {name: kicker_voltages['t0'][name] * kicker_coeff[name] for name in kicker_names},
    't1': {name: kicker_voltages['t1'][name] * kicker_coeff[name] for name in kicker_names}
}
kicker_amps = {
    't0': {name: 1.0 for name in kicker_names},
    't1': {name: kicker_angles['t1'][name] / kicker_angles['t0'][name] for name in kicker_names}
}

for name in kicker_names:
    node = lattice.getNodeForName(name)
    param_name = 'kx' if name.startswith('IKICKH') else 'ky'
    angle = kicker_angles['t0'][name]
    if name.startswith('IKICKH'):
        node.setParam('kx', angle)
    elif name.startswith('IKICKV'):
        node.setParam('ky', angle)
        
sp = bunch.getSyncParticle()
t0 = 0.0 # [s]
t1 = 0.001 # [s]
for name in kicker_names:
    t0amp, t1amp = kicker_amps['t0'][name], kicker_amps['t1'][name]
    waveform = SquareRootWaveform(sp, latt_length, t0, t1, t0amp, t1amp)
    lattice.setTimeDepNode(''.join([name, '_1']), waveform)


# Add injection node
#------------------------------------------------------------------------------
xmin = xcenterpos - 0.0085
xmax = xcenterpos + 0.0085
ymin = ycenterpos - 0.0080
ymax = ycenterpos + 0.100

nparts = macros_per_turn
inject_params = (xmin, xmax, ymin, ymax)
inject_node = TeapotInjectionNode(macros_per_turn, bunch, lostbunch, 
                                  inject_params, xFunc, yFunc, lFunc)
addTeapotInjectionNode(lattice, 0.0, inject_node)


# Track
#------------------------------------------------------------------------------
bunch_monitor_node = AnalysisNode(0.0, 'bunch_monitor')
hf.add_node_at_start(lattice, bunch_monitor_node)
        
lattice.setTurns(100)
lattice.trackBunchTurns(bunch, params_dict)

coords = bunch_monitor_node.get_data('bunch_coords', 'all_turns')
for i, X in enumerate(tqdm(coords)):
    np.save('_output/data/X_{}.npy'.format(i), X)