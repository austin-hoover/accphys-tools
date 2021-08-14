from __future__ import print_function
import os

import numpy as np
import scipy.optimize as opt
from scipy.constants import speed_of_light

from bunch import Bunch
from orbit.matrix_lattice import BaseMATRIX, MATRIX_Lattice
from orbit_utils import Matrix
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf


# Global variables
IND_QUAD_NAMES = ['q02', 'q03', 'q04', 'q05', 'q06', 'q12', 'q13',
                  'q14', 'q15', 'q16', 'q17', 'q18', 'q19']
FIELD_LB = np.array([0, -4.35, 0, -7.95, 0, 0, -5.53,
                          0, -4.35, 0, -4.35, 0, -5.53])
FIELD_UB = np.array([5.5, 0, 5.5, 0, 7.95, 5.53, 
                          0, 4.35, 0, 4.35, 0, 5.53, 0])
SHARED_POWER = {
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


def set_rtbt_quad_strengths(lattice, quad_names, quad_strengths):
    for node_name, kq in zip(quad_names, quad_strengths):
        node = lattice.getNodeForName(node_name)
        node.setParam('kq', kq)
        if node_name in SHARED_POWER:
            for dep_node_name in SHARED_POWER[node_name]:
                dep_node = lattice.getNodeForName(dep_node_name)
                dep_node.setParam('kq', kq) 
        
        
class PhaseController:
    """Class to control phases at wire-scanner.
    
    Attributes
    ----------
    lattice : TEAPOT_Lattice
        The lattice to track with.
    matlat : TEAPOT_MATRIX_Lattice
        Linear matrix representation of the lattice.
    init_twiss : dict
        The Twiss parameters at the lattice entrance: {'alpha_x', 'alpha_y', 
        'beta_x', 'beta_y'}
    tracked_twiss : ndarray, shape (nsteps, 6)
        Twiss parameters tracked through the lattice. Columns are 
        [position, phase_x, phase_y, alpha_x, alpha_y, beta_x, beta_y]. The
        phases are normalized by 2pi.
    """
    def __init__(self, lattice, init_twiss, mass, kin_energy, ref_ws_name='ws24'):
        self.lattice = lattice
        self.init_twiss = init_twiss
        self.mass = mass
        self.kin_energy = kin_energy
        self.ref_ws_name = ref_ws_name
        self.sync_matrix_lattice()
        self.tracked_twiss = None
        self.ind_quad_names = IND_QUAD_NAMES
        self.ind_quad_nodes = [lattice.getNodeForName(name) for name in self.ind_quad_names]
        self.default_quad_strengths = self.quad_strengths(self.ind_quad_names)
        self.track()
        self.ref_ws_node = self.lattice.getNodeForName(ref_ws_name)
        self.ref_ws_index = self.node_index(ref_ws_name)
        
    def sync_matrix_lattice(self):
        """Return TEAPOT_MATRIX_Lattice from the TEAPOT_Lattice."""
        bunch, params_dict = hf.initialize_bunch(self.mass, self.kin_energy)
        self.matlat = TEAPOT_MATRIX_Lattice(self.lattice, bunch)
                
    def track(self):
        """Track twiss parameters through the lattice."""
        alpha_x0 = self.init_twiss['alpha_x']
        alpha_y0 = self.init_twiss['alpha_y']
        beta_x0 = self.init_twiss['beta_x']
        beta_y0 = self.init_twiss['beta_y']
        tracked_twiss_x = self.matlat.trackTwissData(alpha_x0, beta_x0, direction='x')
        tracked_twiss_y = self.matlat.trackTwissData(alpha_y0, beta_y0, direction='y')
        pos, nu_x, alpha_x, beta_x = unpack(tracked_twiss_x)
        pos, nu_y, alpha_y, beta_y = unpack(tracked_twiss_y)
        self.tracked_twiss = np.vstack([pos, nu_x, nu_y, alpha_x, alpha_y, beta_x, beta_y]).T
    
    def node_position(self, node_name):
        """Return position of node entrance [m]."""
        node = self.lattice.getNodeForName(node_name)
        return self.lattice.getNodePositionsDict()[node][0]
            
    def node_index(self, node_name, tol=1e-5):
        """Return index of node in array returned by `track`."""
        position = self.node_position(node_name)
        dist_from_node = np.abs(self.tracked_twiss[:, 0] - position)
        return int(np.where(dist_from_node < tol)[0])
    
    def quad_strength(self, quad_name):
        node = self.lattice.getNodeForName(quad_name)
        return node.getParam('kq')
    
    def quad_strengths(self, quad_names):
        return np.array([self.quad_strength(quad_name) for quad_name in quad_names])
    
    def set_quad_strength(self, quad_name, quad_strength):
        """Set independent quad strengths and update the matrix lattice."""
        node = self.lattice.getNodeForName(quad_name)
        node.setParam('kq', quad_strength)
        if quad_name in SHARED_POWER:
            for dep_quad_name in SHARED_POWER[quad_name]:
                self.set_quad_strength(dep_quad_name, quad_strength)
                
    def set_quad_strengths(self, quad_names, quad_strengths):
        for quad_name, quad_strength in zip(quad_names, quad_strengths):
            self.set_quad_strength(quad_name, quad_strength)
        self.sync_matrix_lattice()
        
    def transfer_matrix(self, start_node_name=None, stop_node_name=None):
        """Calculate linear transfer matrix between two nodes."""
        matrix_nodes = self.matlat.getNodes()   
        
        if start_node_name is None:
            start_node_name = matrix_nodes[0].getName()
        if stop_node_name is None:
            stop_node_name = self.ref_ws_name
            
        def index(node_name):
            for i, node in enumerate(matrix_nodes):
                if node.getName().startswith(node_name):
                    return i        
        
        start_index = index(start_node_name)
        stop_index = index(stop_node_name)
        reverse = start_index > stop_index
        
        M = Matrix(7, 7)
        M.unit()
        for node in matrix_nodes[start_index:stop_index]:
            if isinstance(node, BaseMATRIX):
                M = node.getMatrix().mult(M)
        M = [[M.get(i, j) for j in range(4)] for i in range(4)]
        M = np.array(M)
        if reverse:
            M = np.linalg.inv(M)
        return M
    
    def phase_adv(self, node_name):
        """Return phases (divided by 2pi) from lattice entrance to node."""
        return self.tracked_twiss[self.node_index(node_name), [1, 2]]  
    
    def set_phase_adv(self, node_name, nux, nuy, beta_lims=(40., 40.), **lsq_kws):
        """Set phase advance from lattice entrance to the node."""
        ind_quad_names = ['q18', 'q19']
        target_phases = [nux, nuy]
        
        def cost_func(quad_strengths):
            self.set_quad_strengths(ind_quad_names, quad_strengths)
            self.track()
            calc_phases = self.phase_adv(node_name)
            residuals = np.subtract(target_phases, calc_phases)
            cost = np.sum((residuals)**2)
            cost += np.sum(np.clip(self.max_betas()  - beta_lims, 0., None)**2)
            return cost

        idx = [IND_QUAD_NAMES.index(name) for name in ind_quad_names]
        Brho = hf.get_Brho(self.mass, self.kin_energy)
        lb = FIELD_LB[idx] / Brho
        ub = FIELD_UB[idx] / Brho
        beta_lims = np.array(beta_lims)
        guess = self.default_quad_strengths[idx]
        result = opt.least_squares(cost_func, guess, bounds=(lb, ub), **lsq_kws)
        self.set_quad_strengths(ind_quad_names, result.x)
        max_betas = self.max_betas()
        if np.any(max_betas > beta_lims):
            print('WARNING: maximum beta functions exceed limit.')
            print('Max betas =', self.max_betas())
        return result.x
        
    def max_betas(self):
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
        nux0, nuy0 = self.phase_adv(self.ref_ws_name)
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
        return np.array(phases)