import sys
import numpy as np
from scipy import optimize as opt
from tqdm import trange

from bunch import Bunch
from orbit.matching import Optics
from orbit.matching import EnvelopeSolver
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice


class Matcher:
    """Convenience class to find the matched KV envelope.
    
    Attributes
    ----------
    lattice : TEAPOT_Lattice
        The lattice to track with.
    kin_energy : float
        Kinetic energy per particle [GeV].
    eps_x, eps_y : float
        Rms x and y emittances [m rad].
    solver : EnvelopeSolver
        Object that performs the matching.
    matched_params : ndarray
        Gives following parameters as function of positions s: [rx, rxp, ry,
        ryp, Dx, Dxp, s].
    """
    def __init__(self, lattice, kin_energy, eps_x, eps_y):
        self.eps_x = eps_x
        self.eps_y = eps_y
        self.sigma_p = 0.0
        bunch = Bunch()
        bunch.getSyncParticle().kinEnergy(kin_energy)
        self.solver = EnvelopeSolver(Optics().readtwiss_teapot(lattice, bunch))
        
    def match(self, perveance):
        """Find the matched beam for a given perveance."""
        self.matched_params = self.solver.match_root(
            self.eps_x, self.eps_y, self.sigma_p, perveance)
        
    def twiss(self):
        """Return matched Twiss parameters at lattice entrance."""
        rx, rxp, ry, ryp, Dx, Dxp, s = self.matched_params
        beta_x = rx[0]**2 / self.eps_x
        beta_y = ry[0]**2 / self.eps_y
        alpha_x = -rxp[0] * rx[0] / self.eps_x
        alpha_y = -ryp[0] * ry[0] / self.eps_y
        return alpha_x, alpha_y, beta_x, beta_y
        
    def tunes(self):
        """Return depressed tunes of matched beam."""
        rx, rxp, ry, ryp, Dx, Dxp, s = self.matched_params
        mu_x, mu_y = self.solver.phase_advance(rx, ry, Dx, self.eps_x, self.eps_y, self.sigma_p, s)  
        return np.degrees([mu_x, mu_y])
    
    def set_tunes(self, mu_x, mu_y, **kws):
        """Set depressed tunes by varying space charge strength."""
        def cost(perveance):
            self.match(perveance)
            return np.subtract([mu_x, mu_y], self.tunes())
        
        guess = 0.0
        result = opt.least_squares(cost, guess, **kws)
        perveance = result.x[0]
        return perveance
    
    def track(self, perveance, n_turns):
        """Return period-by-period x and y beam sizes."""
        rx, rxp, ry, ryp, Dx, Dxp, s = self.matched_params
        sizes = []
        sizes.append([rx[0], ry[0]])
        sizes.append([rx[-1], ry[-1]])
        for i in trange(n_turns):
            rx, rxp, ry, ryp, Dx, Dxp, s = self.solver.envelope_odeint(
                self.eps_x, self.eps_y, self.sigma_p, perveance, rx[-1], rxp[-1], ry[-1], ryp[-1], Dx[-1], Dxp[-1])
            sizes.append([rx[-1], ry[-1]])
        return np.array(sizes)
