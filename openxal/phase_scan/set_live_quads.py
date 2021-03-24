"""
Set the live quadrupole strengths in the RTBT, then read back
their values to verify it worked.
"""
from xal.ca import ChannelFactory
from xal.tools.beam import Twiss

from lib.phase_controller import PhaseController
from lib.utils import loadRTBT, write_traj_to_file
from lib.utils import init_twiss, design_betas_at_target

# Load RTBT sequence
sequence = loadRTBT()

# Create phase controller
ref_ws_id = 'RTBT_Diag:WS24'
controller = PhaseController(sequence, ref_ws_id)

# Twiss parameters at RTBT entrance
emittance = 20e-6 # arbitrary [m*rad] 
twissX = Twiss(init_twiss['ax'], init_twiss['bx'], emittance)
twissY = Twiss(init_twiss['ay'], init_twiss['by'], emittance)
controller.set_twiss(twissX, twissY)

# Set live quad strengths
quad_id = 'RTBT_Mag:QH03'
caf = ChannelFactory.defaultFactory()
chB = caf.getChannel(quad_id + ':B_Set')
chBook = caf.getChannel(quad_id + ':B_Book')