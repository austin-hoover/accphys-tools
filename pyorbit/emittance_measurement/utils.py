import sys
import time
import fileinput
import subprocess

import numpy as np
import scipy.optimize as opt
from scipy.constants import speed_of_light

from bunch import Bunch
from orbit.analysis import AnalysisNode, WireScannerNode
from orbit.matrix_lattice import BaseMATRIX, MATRIX_Lattice
from orbit_utils import Matrix
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf


# Global variables
rtbt_ind_quad_names = ['q02', 'q03', 'q04', 'q05', 'q06', 'q12', 'q13',
                       'q14', 'q15', 'q16', 'q17', 'q18', 'q19']

rtbt_quad_coeff_lb = np.array([0, -4.35, 0, -7.95, 0, 0, -5.53,
                               0, -4.35, 0, -4.35, 0, -5.53])
rtbt_quad_coeff_ub = np.array([5.5, 0, 5.5, 0, 7.95, 5.53, 
                               0, 4.35, 0, 4.35, 0, 5.53, 0])

shared_power = {
    'q05': ['q07', 'q09', 'q11'],
    'q06': ['q08', 'q10'],
    'q18': ['q20', 'q22', 'q24'],
    'q19': ['q21', 'q23', 'q25'],
}
    
        
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
    positions, nu_arr, alpha_arr, beta_arr : ndarray,
        Arrays of the parameters at each position.
    """
    data = [[s, nu, alpha, beta] 
            for (s, nu), (s, alpha), (s, beta) in zip(*tracked_twiss)]
    return np.array(data).T


def set_rtbt_ind_quad_strengths(lattice, quad_strengths):
    quad_nodes = [lattice.getNodeForName(name) for name in rtbt_ind_quad_names]
    for node, kq in zip(quad_nodes, quad_strengths):
        node.setParam('kq', kq)

    # Handle shared power supplies
    def set_strengths(names, kq):
        for name in names:
            node = lattice.getNodeForName(name)
            node.setParam('kq', kq)
      
    (k02, k03, k04, k05, k06, k12, 
     k13, k14, k15, k16, k17, k18, k19) = quad_strengths
    set_strengths(['q07', 'q09', 'q11'], k05)
    set_strengths(['q08', 'q10'], k06)
    set_strengths(['q20', 'q22', 'q24'], k18)
    set_strengths(['q21', 'q23', 'q25'], k19)
    
        
        
class PhaseController:
    """Class to control phases at wire-scanner.
    
    Attributes
    ----------
    lattice : TEAPOT_Lattice
        The lattice to track with.
    matlat : TEAPOT_MATRIX_Lattice
        Linear matrix representation of the lattice.
    init_twiss : (ax, ay, bx, by)
        The chosen Twiss parameters at the lattice entrance.
    tracked_twiss : ndarray, shape (nsteps, 6)
        Twiss parameters tracked through the lattice. Columns are 
        [position, phase_x, phase_y, alpha_x, alpha_y, beta_x, beta_y]. The
        phases are normalized by 2pi.
    """
    def __init__(self, lattice, init_twiss, mass, kin_energy, ref_ws_name):
        self.lattice = lattice
        self.init_twiss = init_twiss
        self.mass = mass
        self.kin_energy = kin_energy
        self.ref_ws_name = ref_ws_name
        self.matlat = self.get_matrix_lattice()
        self.tracked_twiss = None
        self.ind_quad_nodes = [lattice.getNodeForName(name) for name in rtbt_ind_quad_names]
        self.default_quad_strengths = self.get_quad_strengths()
        self.track_twiss()
        self.ref_ws_node = self.lattice.getNodeForName(ref_ws_name)
        self.ref_ws_index = self.get_node_index(ref_ws_name)
        
    def get_matrix_lattice(self):
        """Return TEAPOT_MATRIX_Lattice from the TEAPOT_Lattice.
        
        This is very slow; the matrices are not stored, but are instead 
        calculated at each step using the method in 
        `/src/teapot/MatrixGenerator.cc`. This is unfortunate since this 
        needs to be called many times when optimizing the quad strengths.
        """
        bunch, params_dict = hf.initialize_bunch(self.mass, self.kin_energy)
        return TEAPOT_MATRIX_Lattice(self.lattice, bunch)
                
    def track_twiss(self):
        """Track twiss parameters through the lattice."""
        alpha_x0, alpha_y0, beta_x0, beta_y0 = self.init_twiss
        tracked_twiss_x = self.matlat.trackTwissData(alpha_x0, beta_x0, direction='x')
        tracked_twiss_y = self.matlat.trackTwissData(alpha_y0, beta_y0, direction='y')
        pos, nu_x, alpha_x, beta_x = unpack(tracked_twiss_x)
        pos, nu_y, alpha_y, beta_y = unpack(tracked_twiss_y)
        self.tracked_twiss = np.vstack([pos, nu_x, nu_y, alpha_x, alpha_y, beta_x, beta_y]).T
    
    def get_node_position(self, node_name):
        """Return position of node entrance [m]."""
        node = self.lattice.getNodeForName(node_name)
        return self.lattice.getNodePositionsDict()[node][0]
            
    def get_node_index(self, node_name, tol=1e-5):
        """Return index of node in array returned by `track_twiss`."""
        position = self.get_node_position(node_name)
        dist_from_node = np.abs(self.tracked_twiss[:, 0] - position)
        return int(np.where(dist_from_node < tol)[0])
            
    def get_quad_strengths(self):
        """Get current independent quad strengths."""
        return [node.getParam('kq') for node in self.ind_quad_nodes]
    
    def set_quad_strengths(self, quad_strengths):
        """Set independent quad strengths and update the matrix lattice."""
        set_rtbt_ind_quad_strengths(self.lattice, quad_strengths)
        self.matlat = self.get_matrix_lattice()
        
    def apply_settings(self, lattice):
        """Adjust quad strengths in `lattice` to current controller state."""
        set_rtbt_ind_quad_strengths(lattice, self.get_quad_strengths())
        
    def get_transfer_matrix(self, node_name):
        """Calculate linear transfer matrix up to a certain node."""
        matrix = Matrix(7, 7)
        matrix.unit()
        for matrix_node in self.matlat.getNodes():
            if matrix_node.getName().startswith(node_name):
                break
            if isinstance(matrix_node, BaseMATRIX):
                matrix = matrix_node.getMatrix().mult(matrix)
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                M[i, j] = matrix.get(i, j)
        return M  
    
    def get_phase_adv(self, start_node_name=None, stop_node_name=None):
        """Return phases (divided by 2pi) at a certain node."""
        if stop_node_name is None:
            stop_node_name = self.ref_ws_id
        if start_node_name is None:
            start_node_name = self.lattice.getNodes()[0].getName()
        start_index = self.get_node_index(start_node_name)
        stop_index = self.get_node_index(stop_node_name)
        start_phases = self.tracked_twiss[start_index, [1, 2]]    
        stop_phases = self.tracked_twiss[stop_index, [1, 2]]    
        return stop_phases - start_phases
    
    def set_phase_adv(self, start_node_name, stop_node_name, nux, nuy, 
                      max_betas=(40., 40.), **kws):
        
        if stop_node_name is None:
            stop_node_name = self.ref_ws_id
        if start_node_name is None:
            start_node_name = self.lattice.getNodes()[0].getName()
        
        nodes = self.lattice.getNodes()
        node_names = [node.getName() for node in nodes]
        start_node_index = node_names.index(start_node_name)
        stop_node_index = node_names.index(stop_node_name)
        nodes_of_interest = nodes[start_node_index : stop_node_index]
        quad_start_index, quad_stop_index = None, None
        for node in nodes_of_interest:
            if node.getName() in rtbt_ind_quad_names:
                quad_start_index = rtbt_ind_quad_names.index(node.getName())
                break
        for node in reversed(nodes_of_interest):
            if node.getName() in rtbt_ind_quad_names:
                quad_stop_index = rtbt_ind_quad_names.index(node.getName())
                break
        ind_quad_names = rtbt_ind_quad_names[quad_start_index : quad_stop_index + 1]
        ind_quad_nodes = [self.lattice.getNodeForName(name) for name in ind_quad_names]
        
        def set_quad_strengths(quad_strengths):
            for node, kq in zip(ind_quad_nodes, quad_strengths):
                node.setParam('kq', kq)
                if node.getName() in shared_power:
                    for dep_node_name in shared_power[node.getName()]:
                        dep_node = self.lattice.getNodeForName(dep_node_name)
                        dep_node.setParam('kq', kq) 
        
        def cost(quad_strengths):
            set_quad_strengths(quad_strengths)
            self.matlat = self.get_matrix_lattice()
            self.track_twiss()
            nux_calc, nuy_calc = self.get_phase_adv(start_node_name, stop_node_name)
            residuals = np.array([nux_calc - nux, nuy_calc - nuy])
            max_betas_calc = self.get_max_betas() 
            penalty = np.clip(max_betas_calc - max_betas, 0.0, None)
            return residuals + penalty
        
        
        Brho = hf.get_Brho(self.mass, self.kin_energy)
        lb = rtbt_quad_coeff_lb[quad_start_index : quad_stop_index + 1] / Brho
        ub = rtbt_quad_coeff_ub[quad_start_index : quad_stop_index + 1] / Brho
        max_betas = np.array(max_betas)
        default_quad_strengths = self.default_quad_strengths[quad_start_index : quad_stop_index + 1]
        
        result = opt.least_squares(cost, default_quad_strengths, bounds=(lb, ub), **kws)
        set_quad_strengths(result.x)
        if np.any(np.array(self.get_max_betas()) > max_betas):
            print 'WARNING: maximum beta functions exceed limit.'
            print 'Max betas =', self.get_max_betas()
        return result.x
        
        

    def set_ref_ws_phases(self, nux, nuy, max_betas=(40., 40.), **kws):
        """Set phases (divided by 2pi) at reference wire-scanner.
        
        The constraint that the beta functions not be too large before the
        reference wire-scanner is added "by hand". We add a penalty to the
        cost function which scales with the severity of the constraint 
        violation.
        
        Parameters
        ----------
        nux, nuy : float
            Desired phases (divided by 2pi).
        max_betas : (max_beta_x, max_beta_y)
            Maximum beta functions allowed before reference wire-scanner.
        **kws
            Key word arguments for `scipy.optimize.least_squares`.
            
        Returns
        -------
        ndarray, shape (13,)
            The independent quadrupole strengths needed to obtain the 
            desired phases.
        """
        Brho = hf.get_Brho(self.mass, self.kin_energy)
        lb = rtbt_quad_coeff_lb / Brho
        ub = rtbt_quad_coeff_ub / Brho
        max_betas = np.array(max_betas)
        
        def cost(quad_strengths):            
            self.set_quad_strengths(quad_strengths)
            self.track_twiss()
            nux_calc, nuy_calc = self.get_ref_ws_phases()
            residuals = np.array([nux_calc - nux, nuy_calc - nuy])
            max_betas_calc = self.get_max_betas() 
            penalty = np.clip(max_betas_calc - max_betas, 0.0, None)
            return residuals + penalty

        result = opt.least_squares(cost, self.default_quad_strengths, 
                                   bounds=(lb, ub), **kws)
        self.set_quad_strengths(result.x)
        if np.any(np.array(self.get_max_betas()) > max_betas):
            print 'WARNING: maximum beta functions exceed limit.'
            print 'Max betas =', self.get_max_betas()
        return result.x
        
    def get_max_betas(self):
        """Get maximum (beta_x, beta_y) between s=0 and reference wire-scanner."""
        return np.max(self.tracked_twiss[:self.ref_ws_index, 5:], axis=0)
    
    def get_phases_for_scan(self, phase_coverage, steps_per_dim, method=2):
        """Return list of phases for scan. 
        
        phase_coverage : float
            Number of degrees to cover in the scan.
        steps_per_dims : int
            Number of steps to take in each dimension.
        method : {1, 2}
            Method 1 varies x with y fixed, then y with x fixed. Method 2
            varies both at the same time.
        """
        total_steps = 2 * steps_per_dim
        nux0, nuy0 = self.get_ref_ws_phases()
        window = 0.5 * phase_coverage / 360
       
        if method == 1:
            delta_nu_list = np.linspace(-window, window, steps_per_dim)
            phases = []
            for delta_nu in delta_nu_list:
                phases.append([nux0 + delta_nu, nuy0])
            for delta_nu in delta_nu_list:
                phases.append([nux0, nuy0 + delta_nu])
            
        elif method == 2:
            nux_list = np.linspace(nux0 - window, nux0 + window, 2 * steps_per_dim)
            nuy_list = np.linspace(nuy0 + window, nuy0 - window, 2 * steps_per_dim)
            phases = list(zip(nux_list, nuy_list))

        return phases