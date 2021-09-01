from __future__ import print_function
import numpy as np
from scipy import optimize as opt

from bunch import Bunch
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import add_analysis_nodes
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot, TEAPOT_Lattice
from orbit.utils import helper_funcs as hf


def get_part_coords(bunch):
    """Return array of transverse particle coordinates from single-particle bunch."""
    return np.array([bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)])


def track_part(lattice, init_coords, mass, kin_energy):
    """Return coords after tracking single particle through lattice."""
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
    lattice.trackBunch(bunch, params_dict)
    return get_part_coords(bunch)


def get_traj(lattice, init_coords, mass, kin_energy):
    """Return single particle trajectory through lattice."""
    bunch_, params_dict_ = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch_.addParticle(x, xp, y, yp, 0.0, 0.0)
    monitor_nodes = add_analysis_nodes(BunchMonitorNode, lattice, dense=True, 
                                       transverse_only=True, mm_mrad=False)
    lattice.trackBunch(bunch_, params_dict_)
    coords, positions = [], []
    for monitor_node in monitor_nodes:
        coords.append(monitor_node.get_data(turn=0)[0])
        positions.append(monitor_node.position)
        monitor_node.clear_data()
    return np.array(coords), np.array(positions)



class InjRegionController:
    
    def __init__(self, ring, mass, kin_energy):
        self.ring = ring
        self.mass = mass
        self.kin_energy = kin_energy
        
        self.kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                             'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']
        self.kicker_nodes = [ring.getNodeForName(name) for name in self.kicker_names]

        # Maximum injection kicker angles at 1 GeV kinetic energy [mrad]
        self.min_kicker_angles = 1.15 * np.array([0.0, 0.0, -7.13, -7.13, -7.13, -7.13, 0.0, 0.0])
        self.max_kicker_angles = 1.15 * np.array([12.84, 12.84, 0.0, 0.0, 0.0, 0.0, 12.84, 12.84])

        # Scale angles based on actual kinetic energy
        kin_energy_scale_factor = hf.get_pc(mass, 1.0) / hf.get_pc(self.mass, self.kin_energy)
        self.min_kicker_angles *= kin_energy_scale_factor
        self.max_kicker_angles *= kin_energy_scale_factor

        # Convert from mrad to rad
        self.min_kicker_angles *= 1e-3
        self.max_kicker_angles *= 1e-3

        # ARTIFICIALLY INCREASE KICKER LIMITS. 
        artificial_increase_factor = 1.5
        self.min_kicker_angles *= artificial_increase_factor
        self.max_kicker_angles *= artificial_increase_factor

        # Identify horizontal and vertical kickers. PyORBIT doesn't distinguish 
        # between the two.
        self.kicker_idx_x = [0, 2, 5, 7]
        self.kicker_idx_y = [1, 3, 4, 6]
        self.kicker_nodes_x = [self.kicker_nodes[i] for i in self.kicker_idx_x]
        self.kicker_nodes_y = [self.kicker_nodes[i] for i in self.kicker_idx_y]
        self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
        self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
        self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
        self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]

        self.corrector_names = ['dmcv_a09', 'dchv_a10', 'dchv_a13', 'dmcv_b01']
        self.corrector_nodes = [self.ring.getNodeForName(name) for name in self.corrector_names]
        self.max_corrector_angle = 0.005 # [rad]
        self.min_corrector_angle = -self.max_corrector_angle
        self.min_corrector_angles_x = np.full(4, self.min_corrector_angle)
        self.max_corrector_angles_x = np.full(4, self.max_corrector_angle)
        self.min_corrector_angles_y = np.full(4, self.min_corrector_angle)
        self.max_corrector_angles_y = np.full(4, self.max_corrector_angle)
        
    def _set_corrector_angles(self, angles_x=None, angles_y=None):
        """Set the kick strengths [rad] of the dipole correctors in the injection region. 

        Parameters
        ----------
        angles_x : ndarray, shape (2,)
            The horizontal kick angles of the of the two inner magnets.
        angles_y : ndarray, shape (2,)
            The vertical kick angles of the magnets
        """  
        angles_x = np.zeros(2) if angles_x is None else angles_x
        angles_y = np.zeros(4) if angles_y is None else angles_y
        for angle, node in zip(angles_y, self.corrector_nodes):
            node.setParam('ky', angle)
        for angle, node in zip(angles_x, self.corrector_nodes[2:4]):
            node.setParam('kx', angle)
        
    def _set_kicker_angles(self, angles_x=None, angles_y=None):
        """Set kick strengths [rad] of the injection kicker magnets.

        Parameters
        ----------
        angles_x : ndarray, shape (4,)
            The kick angle of the four horizontal kicker magnets.
        angles_y : ndarray, shape (4,)
            The kick angle of the four vertical kicker magnets.
        """
        angles_x = np.zeros(4) if angles_x is None else angles_x
        angles_y = np.zeros(4) if angles_y is None else angles_y
        for angle, node in zip(angles_x, self.kicker_nodes_x):
            node.setParam('kx', angle)
        for angle, node in zip(angles_y, self.kicker_nodes_y):
            node.setParam('ky', angle)
            
                 
    def set_coords_at_foil(self, coords, correctors=False, **solver_kws):

        kicker_angles_x, kicker_angles_y = [], []
        x, xp, y, yp = coords

        # Before foil
        sublattice = hf.get_sublattice(self.ring, 'inj_start', None)
        sublattice.reverseOrder() # track backwards from foil to injection start

        def cost_func(v):
            self._set_kicker_angles(angles_x=[v[0], v[1], 0., 0.],
                                    angles_y=[v[2], v[3], 0., 0.])
            coords_outside_inj = track_part(sublattice, [x, -xp, y, -yp], 
                                            self.mass, self.kin_energy)
            return 1e6 * np.sum(coords_outside_inj**2)

        lb = np.append(self.min_kicker_angles_x[:2], self.min_kicker_angles_y[:2])
        ub = np.append(self.max_kicker_angles_x[:2], self.max_kicker_angles_y[:2])
        result = opt.least_squares(cost_func, np.zeros(4), bounds=(lb, ub), **solver_kws)
        kicker_angles_x.extend(result.x[:2])
        kicker_angles_y.extend(result.x[2:])
    
        # After foil
        sublattice = hf.get_sublattice(self.ring, 'inj_mid', 'inj_end')
        
        def cost_func(v):
            self._set_kicker_angles(angles_x=[0., 0., v[0], v[1]],
                                    angles_y=[0., 0., v[2], v[3]])
            coords_outside_inj = track_part(sublattice, [x, xp, y, yp], 
                                            self.mass, self.kin_energy)
            return 1e6 * np.sum(coords_outside_inj**2)

        lb = np.append(self.min_kicker_angles_x[2:], self.min_kicker_angles_y[2:])
        ub = np.append(self.max_kicker_angles_x[2:], self.max_kicker_angles_y[2:])
        result = opt.least_squares(cost_func, np.zeros(4), bounds=(lb, ub), **solver_kws)
        kicker_angles_x.extend(result.x[:2])
        kicker_angles_y.extend(result.x[2:])
        
        # Set all the correct kicker angles.
        self._set_kicker_angles(kicker_angles_x, kicker_angles_y)
        return self.get_kicker_angles()
        
    def set_kicker_angles(self, angles):
        for angle, node in zip(angles, self.kicker_nodes):
            if node in self.kicker_nodes_x:
                node.setParam('kx', angle)
            elif node in self.kicker_nodes_y:
                node.setParam('ky', angle)
            
    def get_kicker_angles(self):
        kicker_angles = []
        for node in self.kicker_nodes:
            if node in self.kicker_nodes_x:
                kicker_angles.append(node.getParam('kx'))
            elif node in self.kicker_nodes_y:
                kicker_angles.append(node.getParam('ky'))
        return np.array(kicker_angles)