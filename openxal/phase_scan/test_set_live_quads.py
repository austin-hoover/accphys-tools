"""
This script changes the live quadrupole strengths in the RTBT, then reads back
the values to verify that it worked.
"""
from lib.phase_controller import PhaseController, all_quad_ids, ws_ids
from lib.utils import loadRTBT, write_traj_to_file
from lib.utils import init_twiss, design_betas_at_target, delete_files_not_folders
from lib.mathfuncs import radians, multiply


# Setup
#------------------------------------------------------------------------------
delete_files_not_folders('./_output/')

sequence = loadRTBT()

# Create phase controller
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Settings
phase_coverage = radians(180)
scans_per_dim = 6
beta_lims = (40, 40)
beta_lim_after_ws24 = 100


# Test on single quad
#------------------------------------------------------------------------------
quad_id = 'RTBT_Mag:QV03'
fractional_change = 0.05
init_field_strength_model = controller.get_field_strength(quad_id)
init_field_strength_live = controller.get_live_field_strength(quad_id)

target_field_strength = (1 + fractional_change) * init_field_strength
controller.set_field_strength(quad_id, target_field_strength)
controller.update_live_quad(quad_id)

final_field_strength_model = controller.get_field_strength(quad_id)
final_field_strength_live = controller.get_live_field_strength(quad_id)

print 'Init field strength  (model) = {} [T/m]'.format(init_field_strength_model)
print 'Init field strength  (live)  = {} [T/m]'.format(init_field_strength_live)
print 'Final field strength (model) = {} [T/m]'.format(init_field_strength_model)
print 'Final field strength (live)  = {} [T/m]'.format(init_field_strength_live)


# Test on multiple quads
#------------------------------------------------------------------------------
controller.restore_default_optics()
mux, muy = controller.get_ref_ws_phases()
mux += radians(5.0)
controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
controller.set_betas_at_target(design_betas_at_target, beta_lim_after_ws24, verbose=1)

model_field_strengths = controller.get_field_strengths(all_quad_ids)
live_field_strengths = controller.get_live_field_strengths(all_quad_ids)

for k_model, k_live in zip(model_field_strengths, live_field_strengths):
    print 'k_model, k_live = {:.4f}, {:.4f}'.format(k_model, k_live)