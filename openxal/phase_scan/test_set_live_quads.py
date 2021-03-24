"""
This script changes the live quadrupole strengths in the RTBT, then reads back
the values to verify that it worked.
"""
from xal.ca import Channel, ChannelFactory
from xal.smf import AcceleratorSeq 
from xal.tools.beam import Twiss

from lib.phase_controller import PhaseController, rtbt_quad_ids
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
epsx = epsy = 20e-6 # arbitrary [m*rad] 
ax0, ay0, bx0, by0 = [init_twiss[key] for key in ('ax', 'ay', 'bx', 'by')]
controller.set_init_twiss(ax0, ay0, bx0, by0, epsx, epsy)

# Beam size constraints
beta_lims = (40, 40)
beta_lim_after_ws24 = 100


# Test on single quad
#------------------------------------------------------------------------------
quad_id = 'RTBT_Mag:QV03'

init_field_strength_model = controller.get_field_strength(quad_id)
final_field_strength_model = 1.05 * init_field_strength_model

init_field_strength_live = controller.get_live_field_strength(quad_id)
controller.update_live_quad(quad_id)
final_field_strength_live = controller.get_live_field_strength(quad_id)

print 'Quad id =', quad_id
print 'Initial field strength (model) = {} [T/m]'.format(init_field_strength_model)
print 'Initial field strength (live) = {} [T/m]'.format(init_field_strength_live)
print 'Initial field strength (model) = {} [T/m]'.format(final_field_strength_model)
print 'Final field strength (live) = {} [T/m]'.format(final_field_strength_live)


# Test on multiple quads
#------------------------------------------------------------------------------
controller.restore_default_optics()
mux, muy = controller.get_ref_ws_phases()
mux += radians(5.0)

controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
controller.set_betas_at_target(design_betas_at_target, beta_lim_after_ws24, verbose=1)

model_field_strengths = controller.get_field_strengths(rtbt_quad_ids)
live_field_strengths = controller.get_live_field_strengths(rtbt_quad_ids)

for k_model, k_live in zip(model_field_strengths, live_field_strengths):
    print 'k_model, k_live = {:.4f}, {:.4f}'.format(k_model, k_live)
