"""
This script changes the strengths of 18 quadrupoles
"""

# Standard 
import sys
# Third party
import numpy as np
import pandas as pd
from scipy import optimize as opt
# PyORBIT
from bunch import Bunch
from orbit.envelope import Envelope
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf
# Local
sys.path.append('/Users/46h/Research/code/accphys')
from tools.utils import delete_files_not_folders

delete_files_not_folders('_output')
    
    
# Settings
#------------------------------------------------------------------------------

# General
mass = 0.93827231 # GeV/c^2
kin_energy = 1.0 # GeV/c^2
intensity = 1e14

# Lattice
latfile = '_latfiles/rtbt_just_after_Q01.lat'
latseq = 'surv'
fringe = False


# Functions
#------------------------------------------------------------------------------

def unpack(tracked_twiss):
    """Get ndarray from tuple returned by `MATRIX_Lattice.trackTwissData`.
    
    Parameters
    ----------
    tracked_twiss : tuple
       ([(position, phase advance/2/pi) , ...], 
        [(position, alpha), ...],
        [(position, beta) , ...]) 
        
    Returns
    -------
    positions, nu_arr, alpha_arr, beta_arr : ndarray
        Arrays of the parameters at each position.
    """
    data = [[s, nu, alpha, beta] 
            for (s, nu), (s, alpha), (s, beta) in zip(*tracked_twiss)]
    return np.array(data).T


class PhaseScan:
    """Class to linearly scan phase advances at one position.
    
    Attributes
    ----------
    lattice : TEAPOT_Lattice
        Lattice which performs the tracking.
    init_twiss : (alpha_x, alpha_y, beta_x, beta_y)
        Twiss parameters at lattice entrance.
    quads : list[QuadTEAPOT]
        List of quadrupoles to vary during scan.
    ws : MonitorTEAPOT
        Wirescanner at which measurements will be taken.
    mass, kin_energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per particle.
    """
    def __init__(self, lattice, init_twiss, mass, kin_energy):
        self.lattice = lattice
        self.init_twiss = init_twiss
        self.bunch, self.params_dict = hf.initialize_bunch(mass, kin_energy)
        self.create_matrix_lattice()
    
    def create_matrix_lattice(self):
        self.matlat = TEAPOT_MATRIX_Lattice(self.lattice, self.bunch)
        
    def track_twiss(self):
        ax0, ay0, bx0, by0 = self.init_twiss
        tracked_twiss_x = self.matlat.trackTwissData(ax0, bx0, direction='x')
        tracked_twiss_y = self.matlat.trackTwissData(ay0, by0, direction='y')
        pos, nux, ax, bx = unpack(tracked_twiss_x)
        pos, nuy, ay, by = unpack(tracked_twiss_y)
        return np.vstack([pos, nux, nuy, ax, by, bx, by]).T
        
    def set_quads(self, names):
        self.quads = lattice.getNodesForNames(names)
   
    def set_ws(self, name):
        self.ws = lattice.getNodeForName(name)
        self.ws_position = self.lattice.getNodePositionsDict()[self.ws][0]
        twiss = self.track_twiss()
        dist_from_ws = np.abs(twiss[:, 0] - self.ws_position)
        self.ws_idx = int(np.where(dist_from_ws < 1e-5)[0])
        
    def set_quad_strengths(self, quad_strengths):
        for quad, quad_strength in zip(self.quads, quad_strengths):
            quad.setParam('kq', quad_strength)
        self.create_matrix_lattice()
            
    def get_quad_strengths(self):
        # 2, 3, 4, 5, 6, 
        return [quad.getParam('kq') for quad in self.quads]
            
    def get_phases_at_ws(self):
        twiss = self.track_twiss()
        nux, nuy = twiss[self.ws_idx, [1, 2]]
        return nux, nuy
    
    def set_phases_at_ws(self, nux_target, nuy_target):
        
        def error(quad_strengths):
            self.set_quad_strengths(quad_strengths)
            nux, nuy = self.get_phases_at_ws()
            return 1e6 * 0.5 * sum([(nux - nux_target)**2, (nuy - nuy_target)**2])
        
        def constr_func(quad_strengths):
            twiss = self.track_twiss()
            beta_max = np.max(twiss[:78, [5, 6]])
            return beta_max
        
        max_quad_strength = 1.5 # [m-1]
        bounds = opt.Bounds(-max_quad_strength, max_quad_strength)
        constr = opt.NonlinearConstraint(constr_func, 0., 40.)
        result = opt.minimize(error, self.get_quad_strengths(), 
                              bounds=bounds, constraints=constr, method='trust-constr', options={'verbose':2, 'xtol':1e-6})
        self.set_quad_strengths(result.x)

#------------------------------------------------------------------------------

# Initialize
lattice = hf.lattice_from_file(latfile, latseq, fringe)
init_twiss = (-8.082, 4.380, 23.373, 13.455) # (ax, ay, bx, by)
scanner = PhaseScan(lattice, init_twiss, mass, kin_energy)

# Set quadrupoles to vary and wire-scanner for measurements
scanner.set_quads(['q{:02}'.format(i) for i in range(2, 21)])
scanner.set_ws('ws20')
scanner.set_phases_at_ws(2.03, 2.45)

# Save Twiss parameters within the lattice
twiss = scanner.track_twiss()
twiss_df = pd.DataFrame(twiss, columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['mux','muy']]  = 2 * np.pi * twiss_df[['nux','nuy']]
twiss_df.to_csv('_output/twiss.dat', sep=' ', index=False)