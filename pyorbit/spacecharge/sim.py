import sys
import numpy as np
from scipy import optimize as opt
from tqdm import trange

from bunch import Bunch
from spacecharge import SpaceChargeCalc2p5D
from orbit.analysis import AnalysisNode
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import KVDist2D
from orbit.matching import Optics
from orbit.matching import EnvelopeSolver
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf


from orbit.bunch_generators import GaussDist1D


dist = GaussDist1D(cut_off=3.0)

X = []
for i in range(10000):
    x, xp = dist.getCoordinates()
    X.append([x, xp])
np.save('X.npy', X)
    
# # Settings
# #------------------------------------------------------------------------------
# # Lattice
# mu_x0 = 118.0 # horizontal phase advance per cell [deg]
# mu_y0 = 118.0 # vertical phase advance per cell [deg]
# cell_length = 5.0 # [m]
# n_cells = 100

# # Initial bunch
# n_parts = 128000 # number of macro particles
# mass = 0.93827231 # particle mass [GeV/c^2]
# kin_energy = 1.0 # particle kinetic energy [GeV/c^2]
# bunch_length = 150.0 # [m]
# eps_x = 20e-6 # [m rad]
# eps_y = 20e-6 # [m rad]
# mu_x = 90.5 # depressed horizontal phase advance [deg]
# mu_y = 90.5 # depressed vertical phase advance [deg]

# # Space charge solver
# max_solver_spacing = 0.1 # [m]
# min_solver_spacing = 1e-6 # [m]
# gridpts = (128, 128, 1) # (x, y, z)


# # Generate matched KV distribution
# # ------------------------------------------------------------------------------
# class Matcher:
    
#     def __init__(self, lattice, kin_energy, eps_x, eps_y):
#         self.eps_x = eps_x
#         self.eps_y = eps_y
#         self.sigma_p = 0.0
#         bunch = Bunch()
#         bunch.getSyncParticle().kinEnergy(kin_energy)
#         self.solver = EnvelopeSolver(Optics().readtwiss_teapot(lattice, bunch))
        
#     def match(self, perveance):
#         self.matched_params = self.solver.match_root(self.eps_x, self.eps_y, self.sigma_p, perveance)
        
#     def twiss(self):
#         r_x, r_xp, r_y, r_yp, D_x, D_xp, s = self.matched_params
#         beta_x = r_x[0]**2 / self.eps_x
#         beta_y = r_y[0]**2 / self.eps_y
#         alpha_x = -r_xp[0] * r_x[0] / self.eps_x
#         alpha_y = -r_yp[0] * r_y[0] / self.eps_y
#         return alpha_x, alpha_y, beta_x, beta_y
        
#     def phase_adv(self):
#         r_x, r_xp, r_y, r_yp, D_x, D_xp, s = self.matched_params
#         mu_x, mu_y = self.solver.phase_advance(r_x, r_y, D_x, self.eps_x, self.eps_y, self.sigma_p, s)  
#         return np.degrees([mu_x, mu_y])
        

# lattice = hf.fodo_lattice(mu_x0, mu_y0, cell_length, fill_fac=0.5, start='quad')
        
# print 'Setting depressed phase advances.'
# def cost(perveance, mu_x, mu_y, matcher):
#     matcher.match(perveance)
#     return np.subtract([mu_x, mu_y], matcher.phase_adv())

# matcher = Matcher(lattice, kin_energy, eps_x, eps_y)
# result = opt.least_squares(cost, 0.0, args=(mu_x, mu_y, matcher), verbose=2)
# perveance = result.x[0]
# intensity = hf.get_intensity(perveance, mass, kin_energy, bunch_length)
# print 'Matched beam:'
# print '    Perveance = {:.3e}'.format(perveance)
# print '    Intensity = {:.3e}'.format(intensity)
# print '    Bare lattice phase advances:', mu_x0, mu_y0
# print '    Depressed phase advances:', matcher.phase_adv()

# print 'Generating bunch.'
# bunch, params_dict = hf.coasting_beam('gaussian', n_parts, matcher.twiss(), (eps_x, eps_y), 
#                                       bunch_length, mass, kin_energy, intensity, cut_off=3.0)

# # Add space charge nodes
# lattice.split(max_solver_spacing)    
# calc2p5d = SpaceChargeCalc2p5D(*gridpts)
# sc_nodes = setSC2p5DAccNodes(lattice, min_solver_spacing, calc2p5d)

# # Add analysis nodes
# monitor_node = AnalysisNode(0.0, kind='bunch_monitor')
# stats_node = AnalysisNode(0.0, kind='bunch_stats')
# hf.add_node_at_start(lattice, stats_node)
# hf.add_node_at_start(lattice, monitor_node)

# print 'Tracking bunch...'
# hf.track_bunch(bunch, params_dict, lattice, n_cells)

# # Save data
# moments = stats_node.get_data('bunch_moments', 'all_turns')
# np.save('moments.npy', moments)
# coords = monitor_node.get_data('bunch_coords', 'all_turns')
# np.save('coords.npy', coords)