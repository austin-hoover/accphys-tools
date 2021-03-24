"""
This script scans the phases at WS24 in the RTBT in the OpenXAL linear model.
Each phase (horizontal and vertical) is scanned through the range 
[mu - 90 deg, mu + 90 deg]. 
"""

# To do: it's probably best to import everything you are using even though
# it is already imported in the `lib` module.
from xal.tools.beam import Twiss

from lib.phase_controller import PhaseController
from lib.utils import loadRTBT, write_traj_to_file
from lib.utils import init_twiss, design_betas_at_target
from lib.mathfuncs import radians


# Setup
#------------------------------------------------------------------------------
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

# Scan settings
phase_coverage = radians(180)
scans_per_dim = 10
beta_lims = (40, 40)

# Save indices of wire-scanners in trajectory
controller.track()
file = open('_output/ws_index_in_trajectory.dat', 'w')
for i in ['02', '20', '21', '23', '24']:
    ws_id = 'RTBT_Diag:WS' + i
    index = controller.trajectory.indicesForElement(ws_id)[0]
    file.write('name={}, index={}\n'.format(ws_id, index))
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

for i, (mux, muy) in enumerate(phases, start=1):
    
    print 'Scan {}/{}'.format(i, 2 * scans_per_dim)
    print 'Setting phases at {}.'.format(ref_ws_id)
    controller.set_ref_ws_phases(mux, muy, beta_lims, verbose=1)
    print 'Setting betas at target.'
    controller.set_betas_at_target(design_betas_at_target, verbose=1)
    
    # Save optics
    filename = '_output/twiss{}.dat'.format(i)
    write_traj_to_file(controller.get_twiss(), controller.positions, filename)
    
    # Check for violations 
    mux_calc, muy_calc = controller.get_ref_ws_phases()
    max_betas = controller.get_max_betas()
    betas_at_target = controller.get_betas_at_target()
    max_betas_anywhere = controller.get_max_betas(stop_id=None)
    print '  Max betas anywhere:', max_betas_anywhere
    
    if abs(mux - mux_calc) > 0.1 or abs(muy - muy_calc) > 0.1:
        print 'Phases are incorrect.'
        break
    if max_betas[0] > beta_lims[0]:
        print 'Beta_x too large before WS24.'
        break
    if max_betas[1] > beta_lims[1]:
        print 'Beta_y too large before WS24.'
        break
    print ''