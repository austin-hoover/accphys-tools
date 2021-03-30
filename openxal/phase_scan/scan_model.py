"""
This script performs the phase scan using the linear model. For each scan
it writes the files 'transfer_matrix_elems_i.dat' and 'moments_i.dat', where i
is the scan number. Each of row in these files corresponds to a different 
wire-scanner.
"""
from lib.phase_controller import PhaseController, all_quad_ids, ws_ids
from lib.utils import loadRTBT, write_traj_to_file
from lib.utils import init_twiss, design_betas_at_target, delete_files_not_folders
from lib.mathfuncs import radians, multiply


# Setup
#------------------------------------------------------------------------------
delete_files_not_folders('./output/')

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

# Save wire-scanner indices in trajectory (for plotting)
file = open('output/ws_index_in_trajectory.dat', 'w')
for ws_id in ws_ids:
    index = controller.trajectory.indicesForElement(ws_id)[0]
    file.write('name = {}, index = {}\n'.format(ws_id, index))
file.close()


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

for scan_index, (mux, muy) in enumerate(phases, start=1):
    
    print 'Scan {}/{}'.format(scan_index, 2 * scans_per_dim)
    print 'Setting phases at {}.'.format(ref_ws_id)
    controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
    print 'Setting betas at target.'
    controller.set_betas_at_target(design_betas_at_target, beta_lim_after_ws24, verbose=1)
    print '  Max betas anywhere:', controller.get_max_betas(stop_id=None)
    print ''
    
    # Save Twiss vs. position data
    filename = 'output/twiss_{}.dat'.format(scan_index)
    write_traj_to_file(controller.get_twiss(), controller.positions, filename)

    # Save transfer matrix at each wire-scanner. There will be one row per 
    # wire-scanner in the order [ws02, ws20, ws21, ws23, ws24]. Each row lists
    # the 16 elements of the transfer matrix in the order [00, 01, 02, 03, 10,
    # 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33].
    file = open('output/transfer_mat_elems_{}.dat'.format(scan_index),'w')
    fstr = 16 * '{} ' + '\n'
    for ws_id in ws_ids:
        M = controller.get_transfer_matrix_at(ws_id)
        elements = [elem for row in M for elem in row]
        file.write(fstr.format(*elements))
    file.close()

    # Save real space beam moments at each wire-scanner. There will be one row 
    # per wire-scanner in the order [ws02, ws20, ws21, ws23, ws24]. Each row 
    # lists [<xx>, <yy>, <xy>].
    file = open('output/moments_{}.dat'.format(scan_index), 'w')
    for ws_id in ws_ids:
        moments = controller.get_moments_at(ws_id)
        file.write('{} {} {}\n'.format(*moments))
    file.close()