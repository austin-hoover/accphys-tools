# Standard 
import sys
import time
import fileinput
import subprocess
# Third party
import numpy as np
import scipy.optimize as opt
from scipy.constants import speed_of_light
# PyORBIT
from bunch import Bunch
from orbit.analysis import AnalysisNode, WireScannerNode
from orbit.bunch_generators import TwissContainer, GaussDist2D, WaterBagDist2D, KVDist2D
from orbit.envelope import Envelope
from orbit.matrix_lattice import BaseMATRIX, MATRIX_Lattice
from orbit_utils import Matrix
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf

# Global variables
madx_output_files = ['esave', 'madx.ps', 'optics1', 'optics2', 'rtbt.lat']
rtbt_quad_names = ['q02', 'q03', 'q04', 'q05', 'q06', 'q12', 'q13',
                   'q14', 'q15', 'q16', 'q17', 'q18', 'q19']
rtbt_quad_coeff_lb = np.array([0, -5.4775, 0, -7.96585, 0, 0, -7.0425,
                               0, -5.4775, 0, -5.4775, 0, -7.0425])
rtbt_quad_coeff_ub = np.array([5.4775, 0, 7.0425, 0, 7.96585, 7.0425, 
                               0, 5.4775, 0, 5.4775, 0, 7.0425, 0])


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


def get_rtbt_quad_nodes(lattice):
    return [lattice.getNodeForName(name) for name in rtbt_quad_names]


def set_rtbt_quad_strengths(lattice, quad_strengths):
    quad_nodes = get_rtbt_quad_nodes(lattice)
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
        self.mass, self.kin_energy = mass, kin_energy
        self.matlat = self.get_matrix_lattice()
        self.init_twiss = init_twiss
        self.tracked_twiss = None
        self.quad_nodes = get_rtbt_quad_nodes(self.lattice)
        self.default_quad_strengths = self.get_quad_strengths()
        self.track_twiss()
        self.ref_ws_name = ref_ws_name
        self.ref_ws_node = self.lattice.getNodeForName(ref_ws_name)
        self.ref_ws_index = self.get_node_index(ref_ws_name)
        
    def get_matrix_lattice(self):
        """Obtain linear matrix representation of lattice.
        
        This is very slow; the matrices are not stored, but are instead 
        calculated at each step using the method in 
        `/src/teapot/MatrixGenerator.cc`. This is unfortunate since this 
        needs to be called many times when optimizing the quad strengths.
        """
        bunch, params_dict = hf.initialize_bunch(self.mass, self.kin_energy)
        return TEAPOT_MATRIX_Lattice(self.lattice, bunch)
                
    def track_twiss(self):
        """Track twiss parameters through the lattice."""
        ax0, ay0, bx0, by0 = self.init_twiss
        tracked_twiss_x = self.matlat.trackTwissData(ax0, bx0, direction='x')
        tracked_twiss_y = self.matlat.trackTwissData(ay0, by0, direction='y')
        pos, nux, ax, bx = unpack(tracked_twiss_x)
        pos, nuy, ay, by = unpack(tracked_twiss_y)
        self.tracked_twiss = np.vstack([pos, nux, nuy, ax, ay, bx, by]).T
    
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
        return [node.getParam('kq') for node in self.quad_nodes]
    
    def set_quad_strengths(self, quad_strengths):
        """Set quad strengths and update the matrix lattice. 
        
        Only 13 are provided due to shared power supplies. 
        """
        set_rtbt_quad_strengths(self.lattice, quad_strengths)
        self.matlat = self.get_matrix_lattice()
        
    def apply_settings(self, lattice):
        """Adjust quad strengths in `lattice` to reflect the current controller
        state."""
        set_rtbt_quad_strengths(lattice, self.get_quad_strengths())
        
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
    
    def get_phases(self, node_name):
        """Return phases (divided by 2pi) at a certain node."""
        index = self.get_node_index(node_name)
        return self.tracked_twiss[index, [1, 2]]    
        
    def get_ref_ws_phases(self):
        """Return phases (divided by 2pi) at reference wire-scanner."""
        return self.tracked_twiss[self.ref_ws_index, [1, 2]]    

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
        """
        Brho = hf.get_Brho(self.mass, self.kin_energy)
        lb = rtbt_quad_coeff_lb / Brho
        ub = rtbt_quad_coeff_ub / Brho
        
        def cost(quad_strengths):            
            self.set_quad_strengths(quad_strengths)
            self.track_twiss()
            nux_calc, nuy_calc = self.get_ref_ws_phases()
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
        if np.any(np.array(self.get_max_betas()) > max_betas):
            print 'WARNING: maximum beta functions exceed limit.'
        
    def get_max_betas(self):
        """Get maximum (beta_x, beta_y) from s=0 to reference wire-scanner."""
        return np.max(self.tracked_twiss[:self.ref_ws_index, 5:], axis=0)
    
    def get_phases_for_scan(self, phase_coverage, nsteps):
        """Return list of phases for scan. 
        
        First the horizontal phases are scanned, then the vertical.
        """
        nux0, nuy0 = self.get_ref_ws_phases()
        window = 0.5 * phase_coverage
        delta_nu_list = np.linspace(-window, window, nsteps) / 360
        phases = []
        for delta_nu in delta_nu_list:
            phases.append([nux0 + delta_nu, nuy0])
        for delta_nu in delta_nu_list:
            phases.append([nux0, nuy0 + delta_nu]) 
        return phases
    

def get_coord_array(kind, twiss, emittances, nparts, mode=1):
    """Return transverse bunch coordinate array given 2D Twiss parameters.
    
    Parameters
    ----------
    kind : {'danilov', 'kv', 'gaussian', 'waterbag'}
        The kind of distribution.
    init_twiss : (alpha_x, alpha_y, beta_x, beta_y)
        The 2D Twiss parameters.
    emittances : (eps_x, eps_y)
        The r.m.s. emittances.
    nparts : int
        Number of macroparticles.
    mode : {1, 2}
        The rotational mode if the Danilov distribution is chosen.
    """
    ax, ay, bx, by = twiss
    ex, ey = emittances
    if kind == 'danilov':
        intrinsic_emittance = ex + ey
        ex_frac = ex / intrinsic_emittance
        env = Envelope(intrinsic_emittance, mode=mode)
        env.fit_twiss2D(ax, ay, bx, by, ex_frac)
        X = env.generate_dist(nparts)
    else:
        constructors = {'kv':KVDist2D, 
                        'gaussian':GaussDist2D, 
                        'waterbag':WaterBagDist2D}
        (ax, ay, bx, by) = init_twiss
        twissX = TwissContainer(ax, bx, ex)
        twissY = TwissContainer(ay, by, ey)
        kws = {'cut_off':3} if kind == 'gaussian' else {} 
        dist_generator = constructors[kind](twissX, twissY, **kws)
        X = np.array([dist_generator.getCoordinates() for _ in range(nparts)])
    return X