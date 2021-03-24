import math

from xal.model.probe import Probe
from xal.sim.scenario import AlgorithmFactory, ProbeFactory, Scenario
from xal.tools.beam import Twiss, PhaseVector, CovarianceMatrix
from xal.tools.beam.calc import SimpleSimResultsAdaptor, CalculationsOnBeams
from xal.extension.solver import Trial, Variable, Scorer, Stopper, Solver, Problem
from xal.extension.solver.ProblemFactory import getInverseSquareMinimizerProblem
from xal.extension.solver.SolveStopperFactory import maxEvaluationsStopper
from xal.extension.solver.algorithm import SimplexSearchAlgorithm

from mathfuncs import subtract, norm, step_func, put_angle_in_range
from utils import get_trial_vals, solve


# Independent quadrupoles before Q25
quad_ids = ['RTBT_Mag:QH02', 'RTBT_Mag:QV03', 'RTBT_Mag:QH04', 
            'RTBT_Mag:QV05', 'RTBT_Mag:QH06', 'RTBT_Mag:QH12',
            'RTBT_Mag:QV13', 'RTBT_Mag:QH14', 'RTBT_Mag:QV15', 
            'RTBT_Mag:QH16', 'RTBT_Mag:QV17', 'RTBT_Mag:QH18', 
            'RTBT_Mag:QV19']
quad_coeff_lb = [0, -5.4775, 0, -7.96585, 0, 0, -7.0425, 
                 0, -5.4775, 0, -5.4775, 0, -7.0425]
quad_coeff_ub = [5.4775, 0, 7.0425, 0, 7.96585, 7.0425, 
                 0, 5.4775, 0, 5.4775, 0, 7.0425, 0]

# Last four quadrupoles
last_four_quad_ids = ['RTBT_Mag:QV27', 'RTBT_Mag:QH28', 'RTBT_Mag:QV29', 
                      'RTBT_Mag:QH30']
last_four_quad_coeff_lb = [-5.4775, 0, -5.4775, 0] # Don't know if these are correct... using them for now.
last_four_quad_coeff_ub = [0, 5.4775, 0, 5.4775]


class PhaseController:
    """Class to control phases at one wire-scanner in the RTBT."""
    def __init__(self, sequence, ref_ws_id):
        self.sequence = sequence
        self.scenario = Scenario.newScenarioFor(sequence)
        self.algorithm = AlgorithmFactory.createEnvelopeTracker(sequence)
        self.algorithm.setUseSpacecharge(False)
        self.probe = ProbeFactory.getEnvelopeProbe(sequence, self.algorithm)
        self.probe.setBeamCurrent(0.0)
        self.scenario.setProbe(self.probe)
        self.Sigma = CovarianceMatrix()
        self.probe.setCovariance(self.Sigma)
        self.ref_ws_id = ref_ws_id
        self.ref_ws_node = sequence.getNodeWithId(ref_ws_id)
        self.default_field_strengths = self.get_field_strengths()
        self.trajectory = None      

    def set_twiss(self, twissX, twissY):
        """Set initial Twiss parameters."""
        Sigma = CovarianceMatrix().buildCovariance(twissX, twissY, Twiss(0, 0, 0))
        self.probe.setCovariance(Sigma)
        
    def track(self):
        """Return envelope trajectory through the lattice."""
        self.scenario.resetProbe()
        self.scenario.run()
        self.trajectory = self.probe.getTrajectory()
        self.adaptor = SimpleSimResultsAdaptor(self.trajectory) 
        self.states = self.trajectory.getStatesViaIndexer()
        self.positions = [state.getPosition() for state in self.states]
        return self.trajectory
        
    def get_twiss(self):
        """Get Twiss parameters at every state in the trajectory."""
        if self.trajectory is None:
            self.track()
        twiss = []
        for state in self.states:
            twissX, twissY, _ = self.adaptor.computeTwissParameters(state)
            ax, bx = twissX.getAlpha(), twissX.getBeta()
            ay, by = twissY.getAlpha(), twissY.getBeta()
            ex, ey = twissX.getEmittance(), twissY.getEmittance()
            nux, nuy, _ = self.adaptor.computeBetatronPhase(state).toArray()
            twiss.append([nux, nuy, ax, ay, bx, by, ex, ey])
        return twiss
    
    def get_max_betas(self, start_id='RTBT_Mag:QH02', stop_id='RTBT_Diag:WS24'):
        """Get maximum beta functions from `start_id to `stop_id`."""
        if self.trajectory is None:
            self.track()
        lo = self.trajectory.indicesForElement(start_id)[0]
        hi = -1 if stop_id is None else self.trajectory.indicesForElement(stop_id)[-1]
        twiss = self.get_twiss()
        bx_list, by_list = [], []
        for (nux, nuy, ax, ay, bx, by, ex, ey) in twiss[lo:hi]:
            bx_list.append(bx)
            by_list.append(by)
        return max(bx_list), max(by_list)
    
    def get_betas_at_target(self):
        """Get beta functions at the target."""
        if self.trajectory is None:
            self.track()
        twiss = self.get_twiss()
        return twiss[-1][4:6]
    
    def set_field_strength(self, quad_id, field_strength):
        """Set field strength [T/m] of model quad element."""
        node = self.sequence.getNodeWithId(quad_id)
        for elem in self.scenario.elementsMappedTo(node): 
            elem.setMagField(field_strength)
            
    def get_field_strength(self, quad_id):
        """Get field strength [T/m] of model quad element."""
        node = self.sequence.getNodeWithId(quad_id)
        return self.scenario.elementsMappedTo(node)[0].getMagField()
    
    def _set_field_strengths(self, quad_ids, field_strengths):
        """Set field strengths [T/m] of list of model quad elements."""
        if type(field_strengths) is float:
            field_strengths = len(quad_ids) * [field_strengths]
        for quad_id, field_strength in zip(quad_ids, field_strengths):
            self.set_field_strength(quad_id, field_strength)
    
    def set_field_strengths(self, field_strengths): 
        """Set field strengths [T/m] of model quad elements Q03 through Q25.
        
        Note that only 13 strengths are needed as inputs since some magnets 
        share power supplies.
        """
        self._set_field_strengths(quad_ids, field_strengths)            
        (B02, B03, B04, B05, B06, B12,
         B13, B14, B15, B16, B17, B18, B19) = field_strengths  
        self._set_field_strengths(['RTBT_Mag:QV07', 'RTBT_Mag:QV09', 'RTBT_Mag:QV11'], B05)
        self._set_field_strengths(['RTBT_Mag:QH08', 'RTBT_Mag:QH10'], B06)
        self._set_field_strengths(['RTBT_Mag:QH20', 'RTBT_Mag:QH22', 'RTBT_Mag:QH24'], B18)
        self._set_field_strengths(['RTBT_Mag:QV21', 'RTBT_Mag:QV23', 'RTBT_Mag:QV25'], B19)
            
    def set_last_four_field_strengths(field_strenths):
        """Set the field strengths of the last four model quads elements."""
        for quad_id, field_strength in zip(last_four_quad_ids, field_strengths):
            self.controller.set_field_strength(quad_id, field_strength)
            
    def get_field_strengths(self):
        """Get current field strengths [T/m] of the model quad elements.
        
        Note that only 13 strengths are returned since some magnets 
        share power supplies.
        """
        return [self.get_field_strength(quad_id) for quad_id in quad_ids]
    
    def set_live_field_strengths(self):
        """Update the live quads to reflect the current model values."""
        raise NotImplementedError
    
    def set_ref_ws_phases(self, mux, muy, beta_lims=(40, 40), maxiters=1000, 
                          tol=1e-8, verbose=2):
        """Set x and y phases at reference wire-scanner."""        
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.beta_lims = beta_lims
                self.target_phases = [mux, muy]
                
            def score(self, trial, variables):
                field_strengths = get_trial_vals(trial, variables)            
                self.controller.set_field_strengths(field_strengths)
                self.controller.track()
                calc_phases = self.controller.get_ref_ws_phases()
                cost = norm(subtract(calc_phases, self.target_phases))
                cost *= (1 + self.penalty_function())**2
                return cost
            
            def penalty_function(self):
                max_betas = self.controller.get_max_betas() 
                penalty = 0.
                penalty += step_func(max_betas[0] - self.beta_lims[0])
                penalty += step_func(max_betas[1] - self.beta_lims[1])
                return penalty
            
        scorer = MyScorer(self)
        var_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B12', 'B13', 'B14',
                     'B15', 'B16', 'B17', 'B18', 'B19']
        bounds = (quad_coeff_lb, quad_coeff_ub)
        init_field_strengths = self.default_field_strengths      
        self.set_field_strengths(init_field_strengths)
        field_strengths = solve(scorer, init_field_strengths, var_names, bounds, maxiters, tol)
        if verbose > 0:
            print '  Desired phases : {:.3f}, {:.3f}'.format(mux, muy)
            print '  Calc phases    : {:.3f}, {:.3f}'.format(*self.get_ref_ws_phases())
            print '  Max betas (Q03 - WS24): {:.3f}, {:.3f}'.format(*self.get_max_betas())
            print '  Betas at target: {:.3f}, {:.3f}'.format(*self.get_betas_at_target())
        if verbose > 1:
            print solver.getScoreBoard()
        
    def get_ref_ws_phases(self):
        """Return x and y phases (mod 2pi) at reference wire-scanner."""
        if self.trajectory is None:
            self.track()
        ws_state = self.trajectory.statesForElement(self.ref_ws_id)[0]
        ws_phases = self.adaptor.computeBetatronPhase(ws_state)
        return ws_phases.getx(), ws_phases.gety()
    
    def set_betas_at_target(self, betas, maxiters=1000, tol=1e-8, verbose=0,
                            max_beta_ws24_to_target=100.):
        """Vary last 4 quads to set the beta functions at the target.
        
        To do: make sure the beta function isn't too large at any point before the
        target... sometimes it shoots off to very high values.
        """
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.betas = betas
                
            def score(self, trial, variables):
                field_strengths = get_trial_vals(trial, variables)            
                self.controller._set_field_strengths(last_four_quad_ids, field_strengths)
                self.controller.track()
                residuals = subtract(self.betas, self.controller.get_betas_at_target())
                cost = norm(residuals)
                return norm(residuals) + self.penalty_function()**2
                
            def penalty_function(self):
                max_betas = self.controller.get_max_betas(start_id='RTBT_Diag:WS24', stop_id=None)
                penalty = 0.
                penalty += step_func(max_betas[0] - max_beta_ws24_to_target) 
                penalty += step_func(max_betas[1] - max_beta_ws24_to_target)                
                return penalty
            
        scorer = MyScorer(self)
        var_names = ['B27', 'B28', 'B29', 'B30']
        init_field_strengths = [self.get_field_strength(quad_id) 
                                for quad_id in last_four_quad_ids]     
        bounds = (last_four_quad_coeff_lb, last_four_quad_coeff_ub)
        field_strengths = solve(scorer, init_field_strengths, var_names, bounds, maxiters, tol)
        self._set_field_strengths(last_four_quad_ids, field_strengths)
        if verbose > 0:
            print '  Desired betas: {:.3f}, {:.3f}'.format(*betas)
            print '  Calc betas   : {:.3f}, {:.3f}'.format(*self.get_betas_at_target())
        if verbose > 1:
            print solver.getScoreBoard()
            
    def get_phases_for_scan(self, phase_coverage, npts):
        """Create array of phases for scan.

        The phases are centered on the default phase and have a range 
        determined by `phase_coverage`. It is a pain because OpenXAL computes
        the phases mod 2pi.
        """
        # Put phases in range [0, 2pi]
        self.set_field_strengths(self.default_field_strengths)
        mux0, muy0 = self.get_ref_ws_phases()
        
        def _get_phases_for_scan_1d(dim='x'):
            phase = mux0 if dim == 'x' else muy0
            min_phase = put_angle_in_range(phase - 0.5 * phase_coverage)
            max_phase = put_angle_in_range(phase + 0.5 * phase_coverage)
            # Difference between and max phase is always <= 180 degrees
            abs_diff = abs(max_phase - min_phase)
            if abs_diff > math.pi:
                abs_diff = 2*math.pi - abs_diff
            # Return list of phases
            step = abs_diff / (npts - 1)
            phases = [min_phase]
            for _ in range(npts - 1):
                phase = put_angle_in_range(phases[-1] + step)
                phases.append(phase)
            return phases
        
        phases =[]
        for mux in _get_phases_for_scan_1d('x'):
            phases.append([mux, muy0])
        for muy in _get_phases_for_scan_1d('y'):
            phases.append([mux0, muy])
        return phases