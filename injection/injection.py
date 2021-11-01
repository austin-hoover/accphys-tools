"""This module provides control of the SNS injection region."""
from __future__ import print_function
import time

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
    params_dict['lostbunch'] = Bunch() # for apertures
    x, xp, y, yp = init_coords
    bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
    lattice.trackBunch(bunch, params_dict)
    return get_part_coords(bunch)


def get_traj(lattice, init_coords, mass, kin_energy):
    """Return single particle trajectory through lattice."""
    bunch_, params_dict_ = hf.initialize_bunch(mass, kin_energy)
    params_dict_['lostbunch'] = Bunch() # for apertures
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
    
    def __init__(self, ring, mass, kin_energy, dipole_kickers=False):
        self.ring = ring
        self.mass = mass
        self.kin_energy = kin_energy
        
        self.kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                             'ikickv_a12', 'ikickh_a12', 'ikickv_a13', 'ikickh_a13']
        self.kicker_nodes = [ring.getNodeForName(name) for name in self.kicker_names]

        # Maximum injection kicker angles at 1 GeV kinetic energy [mrad]
        self.min_kicker_angles = 1e-3 * np.array([0.0, 0.0, -7.13, -7.13,
                                                  -7.13, -7.13, 0.0, 0.0])
        self.max_kicker_angles = 1e-3 * np.array([12.84, 12.84, 0.0, 0.0, 
                                                  0.0, 0.0, 12.84, 12.84])
        
        # 15% kicker upgrade (from Nick Evans)
        self.min_kicker_angles *= 1.15
        self.max_kicker_angles *= 1.15
        
        if dipole_kickers:
            self.min_kicker_angles = -self.max_kicker_angles

        # Scale angles based on actual kinetic energy.
        self.kin_energy_scale_factor = hf.get_pc(mass, 1.0) / hf.get_pc(self.mass, self.kin_energy)
        self.min_kicker_angles *= self.kin_energy_scale_factor
        self.max_kicker_angles *= self.kin_energy_scale_factor
        
        artificial_kicker_angle_increase_factor = 2.0
        if artificial_kicker_angle_increase_factor != 1.0:
            print('Artificially increasing kicker strength by factor {}'
                  .format(artificial_kicker_angle_increase_factor))
            self.max_kicker_angles *= artificial_kicker_angle_increase_factor
            self.min_kicker_angles *= artificial_kicker_angle_increase_factor

        # Identify horizontal and vertical kickers (PyORBIT doesn't distinguish 
        # between the two).
        self.kicker_idx_x = [0, 2, 5, 7]
        self.kicker_idx_y = [1, 3, 4, 6]
        self.kicker_nodes_x = [self.kicker_nodes[i] for i in self.kicker_idx_x]
        self.kicker_nodes_y = [self.kicker_nodes[i] for i in self.kicker_idx_y]
        self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
        self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
        self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
        self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]

        # Identify corrector dipoles.
        self.corrector_names = ['dmcv_a09', 'dchv_a10', 'dchv_a13', 'dmcv_b01']
        self.corrector_nodes = [self.ring.getNodeForName(name) for name in self.corrector_names]
        self.max_corrector_angle_1GeV = 0.0015 # [rad]
        self.max_corrector_angle = self.kin_energy_scale_factor * self.max_corrector_angle_1GeV
        self.min_corrector_angle = -self.max_corrector_angle
        self.min_corrector_angles_y = np.full(4, self.min_corrector_angle)
        self.max_corrector_angles_y = np.full(4, self.max_corrector_angle)
        
        # Create one sublattice for first half of injection region (before the foil), 
        # and another sublattice for the second half of the injection region (after
        # the foil).
        self.sublattice1 = hf.get_sublattice(self.ring, 'inj_start', None)
        self.sublattice2 = hf.get_sublattice(self.ring, 'inj_mid', 'inj_end')
        
    def _set_kicker_angles(self, angles_x=None, angles_y=None):
        """Set kick strengths [rad] of injection kicker magnets.

        angles_x : ndarray, shape (4,)
            The kick angle of the four horizontal kicker magnets.
        angles_y : ndarray, shape (4,)
            The kick angle of the four vertical kicker magnets.
        """
        if angles_x is not None:
            for angle, node in zip(angles_x, self.kicker_nodes_x):
                node.setParam('kx', angle)
        if angles_y is not None:
            for angle, node in zip(angles_y, self.kicker_nodes_y):
                node.setParam('ky', angle)
                  
    def set_coords_at_foil(self, coords, **solver_kws):
        
        self.sublattice1.reverseOrder() # track backwards from foil to injection start
        x, xp, y, yp = coords
        
        def error():
            coords_start = track_part(self.sublattice1, [x, -xp, y, -yp], 
                                      self.mass, self.kin_energy)
            coords_end = track_part(self.sublattice2, [x, xp, y, yp], 
                                    self.mass, self.kin_energy)
            return 1e6 * (np.sum(coords_start**2) + np.sum(coords_end**2))

        # Horizontal orbit
        def cost_func(v):
            self._set_kicker_angles(angles_x=v)
            return error()
        lb = self.min_kicker_angles_x
        ub = self.max_kicker_angles_x
        opt.least_squares(cost_func, np.zeros(4), bounds=(lb, ub), **solver_kws)
                    
        # Vertical orbit
        def cost_func(v):
            self._set_kicker_angles(angles_y=v)
            return error()
        lb = self.min_kicker_angles_y
        ub = self.max_kicker_angles_y
        opt.least_squares(cost_func, np.zeros(4), bounds=(lb, ub), **solver_kws)
                
        self.sublattice1.reverseOrder()
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
    
    def set_corrector_angles(self, angles_y):
        for angle, node in zip(angles_y, self.corrector_nodes):
            node.setParam('ky', angle)
        
    def get_corrector_angles(self):
        return np.array([node.getParam('ky') for node in self.corrector_nodes])
    
    def bump_vertical_orbit(self, sign_yp=-1, **solver_kws):     
        """Create closed bump in y.
        
        The goal is to make y as positive as possible at the foil and yp as negative 
        as possible at the foil.
        """
        node_pos_dict = self.ring.getNodePositionsDict()
        corrector_positions = [node_pos_dict[node][0] for node in self.corrector_nodes]
        corrector_positions = np.array(corrector_positions)
        ring_length = self.ring.getLength()
       
        def cost_func(corrector_angles):
            self.set_corrector_angles(corrector_angles)
            coords = np.array([0., 0., 0., 0.])
            coords_mid = track_part(self.sublattice1, coords, self.mass, self.kin_energy)
            coords_end = track_part(self.sublattice2, coords_mid, self.mass, self.kin_energy)
            cost = 0.
            cost += (1.0 / (1.0 + 1e3 * coords_mid[2])) # push y as positive
            cost += (1.0 / (1.0 + 1e3 * sign_yp * coords_mid[3])) # push yp as negative
            cost += 1e6 * np.sum(coords_end**2) # enforce closed orbit
            return cost
 
        lb = -self.max_corrector_angle
        ub = +self.max_corrector_angle
        guess = np.zeros(4)
        solver_kws.setdefault('max_nfev', 5000)
        solver_kws.setdefault('verbose', 2)
        result = opt.least_squares(cost_func, guess, bounds=(lb, ub), **solver_kws)    
        
        coords = np.array([0., 0., 0., 0.])
        coords_mid = track_part(self.sublattice1, coords, self.mass, self.kin_energy)
        coords_end = track_part(self.sublattice2, coords_mid, self.mass, self.kin_energy)
        
        coords_mid *= 1000.
        coords_end *= 1000.
        print('Coords at foil with vertical closed bump: ({}, {}, {}, {})'.format(*coords_mid))
        print('Coords at end of injection with vertical closed bump: ({}, {}, {}, {})'.format(*coords_end))
        return result