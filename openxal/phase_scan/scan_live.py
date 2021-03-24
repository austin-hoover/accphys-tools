"""
This script is for performing the phase scan manually.
"""
from lib.phase_controller import PhaseController, rtbt_quad_ids, rtbt_ws_ids
from lib.utils import loadRTBT, write_traj_to_file
from lib.utils import init_twiss, design_betas_at_target
from lib.mathfuncs import radians, multiply


# Setup
#------------------------------------------------------------------------------
# Load RTBT sequence
sequence = loadRTBT()

# Create phase controller
ref_ws_id = 'RTBT_Diag:WS24' # scan phases at this wire-scanner
init_twiss['ex'] = init_twiss['ey'] = 20e-6 # arbitrary [m*rad] 
controller = PhaseController(sequence, ref_ws_id, init_twiss)

# Settings
phase_coverage = radians(180)
scans_per_dim = 5
beta_lims = (40, 40)
beta_lim_after_ws24 = 100
scan_index = 0


# Scan
#------------------------------------------------------------------------------
phases = controller.get_phases_for_scan(phase_coverage, scans_per_dim)
mux0, muy0 = controller.get_ref_ws_phases()

print 'Initial phases at {}: {:.3f}, {:.3f}'.format(ref_ws_id, mux0, muy0)
print 'Phase coverage = {:.3f} rad'.format(phase_coverage)
print 'Scan | mux  | muy [rad]'
print '--------------------------'
for i, (mux, muy) in enumerate(phases, start=1):
    print '{:<4} | {:.2f} | {:.2f}'.format(i, mux, muy)
print ''


print 'Scan index =', scan_index
print 'Setting phases at {}.'.format(ref_ws_id)
controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
print 'Setting betas at target.'
controller.set_betas_at_target(design_betas_at_target, beta_lim_after_ws24, verbose=1)

# Save transfer matrix for all wire-scanners. There will be 5 rows 
# corresponding to the 16 elements of the transfer matrix for each wire-scanner
# in the order [ws02, ws20, ws21, ws23, ws24]. In each row the transfer matrix 
# elements are listed in the order [00, 01, 02, 03, 10, 11, 12, 13, 20, 21, 22,
# 23, 30, 31, 32, 33].
file = open('_output/transfer_matrix_elements_{}.dat'.format(scan_index), 'w')
fstr = 16 * '{} ' + '\n'
for ws_id in rtbt_ws_ids:
    M = controller.get_transfer_matrix_at(ws_id)
    elements = [elem for row in M for elem in row]
    file.write(fstr.format(*elements))
file.close()

# Wire-scanner data then needs to collected externally. Here we use the 
# envelope tracker for comparison. Note that we assume an uncoupled 
# covariance matrix at the RTBT entrance -- for the Danilov distribution this 
# will not be true. Again, there are five rows in the file -- one for each 
# wire-scanner. Each row lists [<xx>, <yy>, <xy>].
file = open('_output/moments_env_{}.dat'.format(scan_index), 'w')
for ws_id in rtbt_ws_ids:
    moments = controller.get_moments_at(ws_id)
    file.write('{} {} {}\n'.format(*moments))