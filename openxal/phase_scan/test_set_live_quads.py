"""
This script changes the live quadrupole strengths in the RTBT, then reads back
the values to verify that it worked.
"""
from xal.ca import Channel, ChannelFactory
from xal.smf import AcceleratorSeq 
from xal.tools.beam import Twiss

from lib.phase_controller import PhaseController
from lib.utils import loadRTBT, write_traj_to_file
from lib.utils import init_twiss, design_betas_at_target
from lib.mathfuncs import radians


# Setup
#------------------------------------------------------------------------------
# Load RTBT sequence and channel factory
sequence = loadRTBT()
channel_factory = ChannelFactory.defaultFactory()

# Create phase controller
ref_ws_id = 'RTBT_Diag:WS24'
controller = PhaseController(sequence, ref_ws_id)

# Twiss parameters at RTBT entrance
emittance = 20e-6 # arbitrary [m*rad] 
twissX = Twiss(init_twiss['ax'], init_twiss['bx'], emittance)
twissY = Twiss(init_twiss['ay'], init_twiss['by'], emittance)
controller.set_twiss(twissX, twissY)

# Beam size constraints
beta_lims = (40, 40)
beta_lim_ws24_to_target = 100


# Test on single quad
#------------------------------------------------------------------------------
quad_id = 'RTBT_Mag:QV03'
ch_Bset = channel_factory.getChannel(quad_id + ':B_Set')
ch_Bbook = channel_factory.getChannel(quad_id + ':B_Book')

init_field_strength_model = controller.get_field_strength(quad_id)
final_field_strength_model = 1.05 * init_field_strength_model

init_field_strength_live = ch_Bset.getValFlt()
controller.update_live_quad(quad_id)
final_field_strength_live = ch_Bset.getValFlt()

print 'Quad id =', quad_id
print 'Initial field strength (model) = {} [T/m]'.format(init_field_strength_model)
print 'Initial field strength (live) = {} [T/m]'.format(init_field_strength_live)
print 'Initial field strength (model) = {} [T/m]'.format(final_field_strength_model)
print 'Final field strength (live) = {} [T/m]'.format(final_field_strength_live)


# Test on multiple quads
#------------------------------------------------------------------------------
controller.set_default_
mux0, muy0 = controller.