# Standard 
import sys
import fileinput
import subprocess
# Third party
import numpy as np
import scipy.optimize as opt
from scipy.constants import speed_of_light
# PyORBIT
from bunch import Bunch
from orbit.analysis import AnalysisNode, WireScannerNode
from orbit.envelope import Envelope
from orbit.matrix_lattice import BaseMATRIX
from orbit_utils import Matrix
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf

#------------------------------------------------------------------------------

madx_output_files = ['esave', 'madx.ps', 'optics1', 'optics2', 'rtbt.lat']


def run(command): 
    """Run bash command."""
    subprocess.call(command, shell=True)
    
    
def run_madx(script, hide_output=True, output_dir='_output/madx/'):
    """Run MADX script and move output."""
    cmd = './madx {} > /dev/null 2>&1' if hide_output else './madx {}'
    run(cmd.format(script))
    run('mv {} {}'.format(' '.join(madx_output_files), output_dir))
    
    
def delete_first_line(file):
    """Delete first line of file."""
    with open(file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(file, 'w') as fout:
        fout.writelines(data[1:])

        
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
        
        
class PhaseScanner:
    """Class to control phases at wire-scanner.
    
    Attributes
    ----------
    lattice : TEAPOT_Lattice
        The lattice to perform the tracking.
    matlat : TEAPOT_MATRIX_Lattice
        Linear matrix representation of the lattice.
    init_twiss : (ax, ay, bx, by)
        The chosen Twiss parameters at the lattice entrance.
    tracked_twiss : ndarray, shape (nsteps, 6)
        Twiss parameters tracked through the lattice. Columns are 
        [position, phase_x, phase_y, alpha_x, alpha_y, beta_x, beta_y]. The
        phases are normalized by 2pi.
    ws_name : str
        Name of wire-scanner at which to measure the phase.
    """
    def __init__(self, teapot_lattice, init_twiss, mass, kin_energy, ws_name):
        self.lattice = teapot_lattice
        self.mass, self.kin_energy = mass, kin_energy
        self.matlat = self.get_matrix_lattice()
        self.init_twiss = init_twiss
        self.tracked_twiss = None
        self.quad_names = ['q02', 'q03', 'q04', 'q05', 'q06', 'q12', 'q13',
                           'q14', 'q15', 'q16', 'q17', 'q18', 'q19']
        self.quad_nodes = [teapot_lattice.getNodeForName(name) 
                           for name in self.quad_names]
        self.default_quad_strengths = self.get_quad_strengths()
        self.ws_name = ws_name
        self.track_twiss()
        self.ws_index = self.get_ws_index(ws_name)
        
    def get_matrix_lattice(self):
        """Get linear matrix represenation of lattice."""
        bunch, params_dict = hf.initialize_bunch(self.mass, self.kin_energy)
        return TEAPOT_MATRIX_Lattice(self.lattice, bunch)
                
    def track_twiss(self):
        """Propagate twiss parameters using matrices."""
        ax0, ay0, bx0, by0 = self.init_twiss
        tracked_twiss_x = self.matlat.trackTwissData(ax0, bx0, direction='x')
        tracked_twiss_y = self.matlat.trackTwissData(ay0, by0, direction='y')
        pos, nux, ax, bx = unpack(tracked_twiss_x)
        pos, nuy, ay, by = unpack(tracked_twiss_y)
        self.tracked_twiss = np.vstack([pos, nux, nuy, ax, ay, bx, by]).T
    
    def get_ws_position(self, ws_name):
        """Return position of wire-scanner."""
        ws_node = self.lattice.getNodeForName(ws_name)
        return self.lattice.getNodePositionsDict()[ws_node][0]
            
    def get_ws_index(self, ws_name):
        """Return index of wire-scanner in array returned by `track_twiss`."""
        ws_position = self.get_ws_position(ws_name)
        dist_from_ws = np.abs(self.tracked_twiss[:, 0] - ws_position)
        ws_index = int(np.where(dist_from_ws < 1e-5)[0])
        return ws_index
            
    def get_quad_strengths(self):
        """Get current quad strengths.
        
        Because of shared power supplies, only 13 strengths are returned.
        """
        return [node.getParam('kq') for node in self.quad_nodes]
    
    def set_quad_strengths(self, quad_strengths):
        """Set quad strengths.
        
        Because of shared power supplies, only 13 strengths need to be 
        provided.
        """
        for node, kq in zip(self.quad_nodes, quad_strengths):
            node.setParam('kq', kq)
            
        # Handle shared power supplies  
        def set_strengths(names, kq):
            for name in names:
                node = self.lattice.getNodeForName(name)
                node.setParam('kq', kq)
                
        (k02, k03, k04, k05, k06, k12, 
         k13, k14, k15, k16, k17, k18, k19) = quad_strengths
        set_strengths(['q07', 'q09', 'q11'], k05)
        set_strengths(['q08', 'q10'], k06)
        set_strengths(['q20', 'q22', 'q24'], k18)
        set_strengths(['q21', 'q23', 'q25'], k19)
        self.matlat = self.get_matrix_lattice()
        
    def get_transfer_matrix_to_ws(self, ws_name):
        """Get transfer matrix from s=0 to wire-scanner."""
        matrix = Matrix(7, 7)
        matrix.unit()
        for matrix_node in self.matlat.getNodes():
            if matrix_node.getName().startswith(ws_name):
                break
            if isinstance(matrix_node, BaseMATRIX):
                matrix = matrix_node.getMatrix().mult(matrix)
        M = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                M[i, j] = matrix.get(i, j)
        return M  
        
    def get_phases_at_ws(self, ws_name):
        """Return phase advances at wirescanner (nux, nuy)."""
        index = self.get_ws_index(ws_name) 
        return self.tracked_twiss[index, [1, 2]]    
        
    def set_phases_at_ws(self, ws_name, nux, nuy, max_betas=(40., 40.), **kws):
        """Set phase advances at wire-scanner.
        
        Using scipy.minimize with constraints included works, but is very slow
        (it might be that the tolerance can be adjusted). So, here we use
        scipy.least_squares and sort of add the constraint by hand. We just
        create a penalty which starts at zero and scales with the severity of 
        the constraint violation. It seems to work and is much faster.
        """
        Brho = hf.get_Brho(self.mass, self.kin_energy)
        lb = (1 / Brho) * np.array([0, -5.4775, 0, -7.96585, 0, 0, -7.0425,
                                    0, -5.4775, 0, -5.4775, 0, -7.0425])
        ub = (1 / Brho) * np.array([5.4775, 0, 7.0425, 0, 7.96585, 7.0425, 
                                    0, 5.4775, 0, 5.4775, 0, 7.0425, 0])

        def cost(quad_strengths):
            self.set_quad_strengths(quad_strengths)
            self.track_twiss()
            nux_calc, nuy_calc = self.get_phases_at_ws(ws_name)
            residuals = np.array([nux_calc - nux, nuy_calc - nuy])
            max_betas_calc = self.get_max_betas() 
            penalty = 0
            penalty += hf.step_func(max_betas_calc[0] - max_betas[0])
            penalty += hf.step_func(max_betas_calc[1] - max_betas[1])
            return residuals * (1 + penalty)

        result = opt.least_squares(cost, self.default_quad_strengths, 
                                   bounds=(lb, ub), **kws)
        new_quad_strengths = result.x
        self.set_quad_strengths(new_quad_strengths)
    
    def get_max_betas(self):
        """Get maximum (beta_x, beta_y) from s=0 to WS24."""
        return np.max(self.tracked_twiss[:self.ws_index, 5:], axis=0)
        
    
def add_analysis_node(lattice, ws_name, kind='env_monitor', mm_mrad=False):
    """Add analysis node as child of MonitorTEAPOT node."""
    ws_node = lattice.getNodeForName(ws_name)
    position = lattice.getNodePositionsDict()[ws_node][0]
    analysis_node = AnalysisNode(position, kind, mm_mrad=mm_mrad)
    ws_node.addChildNode(analysis_node, ws_node.ENTRANCE)
    return analysis_node


def add_ws_node(lattice, nbins, diag_wire_angle, name):
    """Add wirescanner node as chid of MonitorTEAPOT node.

    I made a node called `WireScannerNode` which just returns the <x^2>,
    <y^2>, and <xy> moments. This method adds my custom node onto the 
    already existing MonitorTEAPOT nodes in the lattic which have titles
    like 'ws20'.
    """
    ws_node = lattice.getNodeForName(name)
    my_ws_node = WireScannerNode('my_' + name)
    ws_node.addChildNode(my_ws_node, ws_node.ENTRANCE)
    return my_ws_node