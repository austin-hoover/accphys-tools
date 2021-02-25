# Standard 
import sys
import fileinput
import subprocess
# Third party
import numpy as np
# PyORBIT
from bunch import Bunch
from orbit.analysis import AnalysisNode, WireScannerNode
from orbit.envelope import Envelope
from orbit.matrix_lattice import BaseMATRIX
from orbit_utils import Matrix
from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice
from orbit.utils import helper_funcs as hf


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

    
def get_matlat(lattice, mass, kin_energy):
    """Convert lattice to matrix lattice."""
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    return TEAPOT_MATRIX_Lattice(lattice, bunch)
        

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


class MadxController:
    """Class to interact with MADX scripts.
    
    To do: create backup copy of script.
    
    Attributes
    ----------
    script : str
        Name of MADX script.
    init_twiss : (alpha_x, alpha_y, beta_x, beta_y)
        Twiss parameters at lattice entrance.
    latfile, latseq : str
        Filename and sequence keyword in output lattice file from script.
    """
    def __init__(self, script, init_twiss, latfile='_output/madx/rtbt.lat', latseq='surv'):
        self.script = script
        self.init_twiss = init_twiss
        self.latfile, self.latseq = latfile, latseq
        
    def generate_latfile(self, hide_output=False):
        """Run script to produce lattice file."""
        run_madx(self.script, hide_output)
        delete_first_line(self.latfile) # first line reads: 'none = 0'
    
    def read_latfile(self):
        """Read lattice file into PyORBIT."""
        return hf.lattice_from_file(self.latfile, self.latseq)  
        
    def create_lattice(self, hide_output=False):
        """Run scipt and read lattice file to create TEAPOT_Lattice."""
        self.generate_latfile(hide_output)
        return self.read_latfile()
        
    def set_delta_mu(self, delta_mux=0., delta_muy=0.):
        """Set the phase advances at the wirescanner in the MADX script.
        
        delta_mux and delta_muy are the deviations from the original phase advance.
        """
        for line in fileinput.input(self.script, inplace=True):
            if line.startswith('delta_mux ='):
                line = 'delta_mux = {};\n'.format(delta_mux)
            elif line.startswith('delta_muy ='):
                line = 'delta_muy = {};\n'.format(delta_muy)
            sys.stdout.write(line)
    
    def set_ws(self, name):
        """Set wirescanner to measure with."""
        name = name.upper()
        for line in fileinput.input(self.script, inplace=True):
            if line.startswith('mux_ws = TABLE'):
                line = 'mux_ws = TABLE(TWISS, {}, MUX);\n'.format(name)
            elif line.startswith('muy_ws = TABLE'):
                line = 'muy_ws = TABLE(TWISS, {}, MUY);\n'.format(name)
            elif 'CONSTRAINT' in line and '#S' not in line:
                i = line.find('RANGE')
                line = line.replace(line[i:i+10], 'RANGE={}'.format(name))
            sys.stdout.write(line)
    
    def set_max_beta(self, betax_max, betay_max):
        """Set maximum beta function constraint."""
        for line in fileinput.input(self.script, inplace=True):
            if line.startswith('betax_max ='):
                line = 'betax_max = {};\n'.format(betax_max)
            elif line.startswith('betay_max ='):
                line = 'betay_max = {};\n'.format(betay_max)
            sys.stdout.write(line)
        
        
class Scanner:
    """Class to store TEAPOT lattice and its matrix representation.
    
    Attributes
    ----------
    lattice : TEAPOT_Lattice
        The lattice to perform the tracking.
    matlat : TEAPOT_MATRIX_Lattice
        Linear matrix representation of `lattice`.
    init_twiss : (ax, ay, bx, by)
        The chosen Twiss parameters at the lattice entrance.
    """
    def __init__(self, teapot_lattice, init_twiss, mass, kin_energy):
        self.lattice = teapot_lattice
        self.matlat = get_matlat(teapot_lattice, mass, kin_energy)
        self.init_twiss = init_twiss
        
    def track_twiss(self):
        """Propagate twiss parameters using matrices."""
        ax0, ay0, bx0, by0 = self.init_twiss
        tracked_twiss_x = self.matlat.trackTwissData(ax0, bx0, direction='x')
        tracked_twiss_y = self.matlat.trackTwissData(ay0, by0, direction='y')
        pos, nux, ax, bx = unpack(tracked_twiss_x)
        pos, nuy, ay, by = unpack(tracked_twiss_y)
        return np.vstack([pos, nux, nuy, ax, ay, bx, by]).T
    
    def get_ws_position(self, ws_name):
        """Return position of wire-scanner."""
        ws_node = self.lattice.getNodeForName(ws_name)
        return self.lattice.getNodePositionsDict()[ws_node][0]
            
    def get_ws_index(self, ws_name):
        """Return index of wire-scanner in array returned by `track_twiss`."""
        ws_position = self.get_ws_position(ws_name)
        twiss = self.track_twiss()
        dist_from_ws = np.abs(twiss[:, 0] - ws_position)
        ws_index = int(np.where(dist_from_ws < 1e-5)[0])
        return ws_index
    
    def get_transfer_matrix(self, node_name):
        """Get transfer matrix from s=0 to a given node."""
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
    
    def get_ws_phase(self, ws_name):
        """Return phase advances at the wirescanner."""
        twiss = self.track_twiss()
        nux, nuy = twiss[self.get_ws_index(ws_name), [1, 2]]
        return [nux, nuy]
    
    
def add_analysis_node(lattice, ws_name, kind='env_monitor', mm_mrad=False):
    """Add analysis node as child of MonitorTEAPOT node."""
    ws_node = lattice.getNodeForName(ws_name)
    position = lattice.getNodePositionsDict()[ws_node][0]
    analysis_node = AnalysisNode(position, kind, mm_mrad=mm_mrad)
    ws_node.addChildNode(analysis_node, ws_node.ENTRANCE)
    return analysis_node


def add_ws_node(lattice, ws_name):
    """Add wirescanner node as chid of MonitorTEAPOT node.

    I made a node called `WireScannerNode` which just returns the <x^2>,
    <y^2>, and <xy> moments. This method adds my custom node onto the 
    already existing MonitorTEAPOT nodes in the lattic which have titles
    like 'ws20'.
    """
    ws_node = lattice.getNodeForName(ws_name)
    my_ws_node = WireScannerNode('my_' + ws_name)
    ws_node.addChildNode(my_ws_node, ws_node.ENTRANCE)
    return my_ws_node