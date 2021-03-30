import math

from xal.ca import Channel, ChannelFactory
from xal.smf import Accelerator, AcceleratorSeq 
from xal.model.probe import Probe
from xal.model.probe.traj import Trajectory
from xal.sim.scenario import AlgorithmFactory, ProbeFactory, Scenario
from xal.tools.beam import Twiss, PhaseVector, CovarianceMatrix
from xal.tools.beam.calc import SimpleSimResultsAdaptor, CalculationsOnBeams
from xal.extension.solver import Trial, Variable, Scorer, Stopper, Solver, Problem
from xal.extension.solver.ProblemFactory import getInverseSquareMinimizerProblem
from xal.extension.solver.SolveStopperFactory import maxEvaluationsStopper
from xal.extension.solver.algorithm import SimplexSearchAlgorithm

from mathfuncs import subtract, norm, step_func, put_angle_in_range
from utils import get_trial_vals, minimize, init_twiss

#------------------------------------------------------------------------------
ws_ids = ['RTBT_Diag:WS02', 'RTBT_Diag:WS20', 'RTBT_Diag:WS21', 
          'RTBT_Diag:WS23', 'RTBT_Diag:WS24']
all_quad_ids = [
    'RTBT_Mag:QH02', 'RTBT_Mag:QV03', 'RTBT_Mag:QH04', 'RTBT_Mag:QV05', 
    'RTBT_Mag:QH06', 'RTBT_Mag:QV07', 'RTBT_Mag:QH08', 'RTBT_Mag:QV09', 
    'RTBT_Mag:QH10', 'RTBT_Mag:QV11', 'RTBT_Mag:QH12', 'RTBT_Mag:QV13', 
    'RTBT_Mag:QH14', 'RTBT_Mag:QV15', 'RTBT_Mag:QH16', 'RTBT_Mag:QV17', 
    'RTBT_Mag:QH18', 'RTBT_Mag:QV19', 'RTBT_Mag:QH20', 'RTBT_Mag:QV21', 
    'RTBT_Mag:QH22', 'RTBT_Mag:QV23', 'RTBT_Mag:QH24', 'RTBT_Mag:QV25',    
    'RTBT_Mag:QH26', 'RTBT_Mag:QV27', 'RTBT_Mag:QH28', 'RTBT_Mag:QV29', 
    'RTBT_Mag:QH30']
ind_quad_ids = [ # quads with independent power supplies
    'RTBT_Mag:QH02', 'RTBT_Mag:QV03', 'RTBT_Mag:QH04', 'RTBT_Mag:QV05', 
    'RTBT_Mag:QH06', 'RTBT_Mag:QH12', 'RTBT_Mag:QV13', 'RTBT_Mag:QH14', 
    'RTBT_Mag:QV15', 'RTBT_Mag:QH16', 'RTBT_Mag:QV17', 'RTBT_Mag:QH18', 
    'RTBT_Mag:QV19', 'RTBT_Mag:QH26', 'RTBT_Mag:QV27', 'RTBT_Mag:QH28',
    'RTBT_Mag:QV29', 'RTBT_Mag:QH30']
ind_quad_ids_before_ws24 = ind_quad_ids[:-5]
ind_quad_ids_after_ws24 = ind_quad_ids[-5:]

# Limits on quadrupole field strengths [T/m]
ind_quads_before_ws24_lb = [0, -5.4775, 0, -7.96585, 0, 0, -7.0425, 
                            0, -5.4775, 0, -5.4775, 0, -7.0425]
ind_quads_before_ws24_ub = [5.4775, 0, 7.0425, 0, 7.96585, 7.0425, 
                            0, 5.4775, 0, 5.4775, 0, 7.0425, 0]
ind_quads_after_ws24_lb = [0, -5.4775, 0, -5.4775, 0]    # Don't know if these are correct... using them for now.
ind_quads_after_ws24_ub = [5.4775, 0, 5.4775, 0, 5.4775] # Don't know if these are correct... using them for now.


class PhaseController:
    """Class to control phases at one wire-scanner in the RTBT."""
    def __init__(self, sequence, ref_ws_id, init_twiss):
        """Constructor.
        
        Parameters
        ----------
        sequence : AcceleratorSeq
            Sequence representing the RTBT.
        ref_ws_id : str
            Node id of reference wire-scanner.
        init_twiss : dict
            Dictionary containing 'ax', 'ay', 'bx', 'by', 'ex', 'ey'.
        """
        self.sequence = sequence
        self.scenario = Scenario.newScenarioFor(sequence)
        self.algorithm = AlgorithmFactory.createEnvelopeTracker(sequence)
        self.algorithm.setUseSpacecharge(False)
        self.probe = ProbeFactory.getEnvelopeProbe(sequence, self.algorithm)
        self.probe.setBeamCurrent(0.0)
        self.scenario.setProbe(self.probe)
        self.init_twiss = init_twiss
        self.initialize_envelope()
        self.track()
        self.channel_factory = ChannelFactory.defaultFactory()
        self.ref_ws_id = ref_ws_id
        self.ref_ws_node = sequence.getNodeWithId(ref_ws_id)
        self.default_field_strengths_before_ws24 = self.get_field_strengths(ind_quad_ids_before_ws24)
        self.default_field_strengths_after_ws24 = self.get_field_strengths(ind_quad_ids_after_ws24)

    def initialize_envelope(self):
        """Construct covariance matrix at s=0."""
        self.scenario.resetProbe()
        ax, bx, ex = [self.init_twiss[key] for key in ('ax', 'bx', 'ex')]
        ay, by, ey = [self.init_twiss[key] for key in ('ay', 'by', 'ey')]
        twissX = Twiss(ax, bx, ex)
        twissY = Twiss(ay, by, ey)
        twissZ = Twiss(0, 1, 0)
        Sigma = CovarianceMatrix().buildCovariance(twissX, twissY, twissZ)
        self.probe.setCovariance(Sigma)
        
    def track(self):
        """Return envelope trajectory through the lattice."""
        self.initialize_envelope()
        self.scenario.run()
        self.trajectory = self.probe.getTrajectory()
        self.adaptor = SimpleSimResultsAdaptor(self.trajectory) 
        self.states = self.trajectory.getStatesViaIndexer()
        self.positions = [state.getPosition() for state in self.states]
        return self.trajectory
        
    def get_twiss(self):
        """Get Twiss parameters at every state in the trajectory."""
        twiss = []
        for state in self.states:
            twissX, twissY, _ = self.adaptor.computeTwissParameters(state)
            ax, bx = twissX.getAlpha(), twissX.getBeta()
            ay, by = twissY.getAlpha(), twissY.getBeta()
            ex, ey = twissX.getEmittance(), twissY.getEmittance()
            nux, nuy, _ = self.adaptor.computeBetatronPhase(state).toArray()
            twiss.append([nux, nuy, ax, ay, bx, by, ex, ey])
        return twiss
    
    def get_moments_at(self, node_id):
        """Return tracked [<xx>, <yy>, <xy>] covariances at a certain node."""
        state = self.trajectory.statesForElement(node_id)[0]
        Sigma = state.getCovarianceMatrix()
        return [Sigma.getElem(0, 0), Sigma.getElem(2, 2), Sigma.getElem(0, 2)]
    
    def get_transfer_matrix_at(self, node_id):
        """Return the transfer matrix from s=0 to a certain node."""
        scenario = Scenario.newScenarioFor(self.sequence)
        algorithm = AlgorithmFactory.createTransferMapTracker(self.sequence)
        probe = ProbeFactory.getTransferMapProbe(self.sequence, algorithm)
        scenario.setProbe(probe)
        scenario.run()
        trajectory = probe.getTrajectory()
        states = trajectory.getStatesViaIndexer()
        state = trajectory.statesForElement(node_id)[0]
        transfer_matrix = state.getTransferMap().getFirstOrder()
        M = []
        for i in range(4):
            row = []
            for j in range(4):
                row.append(transfer_matrix.getElem(i, j))
            M.append(row)
        return M
        
    def get_max_betas(self, start_id='RTBT_Mag:QH02', stop_id='RTBT_Diag:WS24'):
        """Get maximum beta functions from `start_id to `stop_id`."""
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
        return self.get_twiss()[-1][4:6]
    
    def set_field_strength(self, quad_id, field_strength):
        """Set field strength [T/m] of model quad."""
        node = self.sequence.getNodeWithId(quad_id)
        for elem in self.scenario.elementsMappedTo(node): 
            elem.setMagField(field_strength)
    
    def set_field_strengths(self, quad_ids, field_strengths):
        """Set field strengths [T/m] of model quads."""
        if type(field_strengths) is float:
            field_strengths = len(quad_ids) * [field_strengths]
        for quad_id, field_strength in zip(quad_ids, field_strengths):
            self.set_field_strength(quad_id, field_strength)
            
        def share_power(dep_quad_ids, indep_quad_id):
            shared_field_strength = self.get_field_strength(indep_quad_id)
            for dep_quad_id in dep_quad_ids:
                self.set_field_strength(dep_quad_id, shared_field_strength)
                
        share_power(['RTBT_Mag:QV07', 'RTBT_Mag:QV09', 'RTBT_Mag:QV11'], 'RTBT_Mag:QV05')
        share_power(['RTBT_Mag:QH08', 'RTBT_Mag:QH10'], 'RTBT_Mag:QH06')
        share_power(['RTBT_Mag:QH20', 'RTBT_Mag:QH22', 'RTBT_Mag:QH24'], 'RTBT_Mag:QH18')
        share_power(['RTBT_Mag:QV21', 'RTBT_Mag:QV23', 'RTBT_Mag:QV25'], 'RTBT_Mag:QV19')         
            
    def get_field_strength(self, quad_id):
        """Get field strength [T/m] of model quad."""
        node = self.sequence.getNodeWithId(quad_id)
        return self.scenario.elementsMappedTo(node)[0].getMagField()
            
    def get_field_strengths(self, quad_ids):
        """Get field strengths [T/m] of model quads."""
        return [self.get_field_strength(quad_id) for quad_id in quad_ids]
    
    def update_live_quad(self, quad_id):
        """Update live quad field strength to reflect the current model value.
        
        UNTESTED
        """
        field_strength = self.get_field_strength(quad_id)
        ch_Bset = self.channel_factory.getChannel(quad_id + ':B_Set')
        ch_Bbook = self.channel_factory.getChannel(quad_id + ':B_Book')
        ch_Bset.putVal(field_strength)
        ch_Bbook.putVal(field_strength)
        
    def update_live_quads(self, quad_ids):
        """Update quad field strengths to reflect the current model values."""
        for quad_id in quad_ids:
            self.update_live_quad(quad_id)   
            
    def get_live_field_strength(self, quad_id):
        """Get field strength [T/m] of live quad.
        
        UNTESTED
        """
        ch_Bset = self.channel_factory.getChannel(quad_id + ':B_Set')
        return ch_Bset.getValFlt()
    
    def get_live_field_strengths(self, quad_ids):
        """Get field strengths [T/m] of live quads."""
        return [get_live_field_strength(quad_id) for quad_id in quad_ids]
    
    def set_ref_ws_phases(self, mux, muy, beta_lims=(40, 40), verbose=0):
        """Set x and y phases at reference wire-scanner.

        Parameters
        ----------
        mux, muy : float
            The desired phase advances at the reference wire-scanner.
        beta_lims : (xmax, ymax)
            Maximum beta functions to allow from s=0 to ws24.
        verbose : int
            If greater than zero, print a before/after summary.
        """        
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.beta_lims = beta_lims
                self.target_phases = [mux, muy]
                
            def score(self, trial, variables):
                field_strengths = get_trial_vals(trial, variables)   
                self.controller.set_field_strengths(ind_quad_ids_before_ws24, field_strengths)
                self.controller.track()
                calc_phases = self.controller.get_ref_ws_phases()
                cost = norm(subtract(calc_phases, self.target_phases))
                return cost + self.penalty_function()**2
            
            def penalty_function(self):
                max_betas = self.controller.get_max_betas() 
                penalty = 0.
                for max_beta, beta_lim in zip(max_betas, self.beta_lims):
                    penalty += step_func(max_beta - beta_lim)
                return penalty
            
        scorer = MyScorer(self)
        var_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B12', 'B13', 'B14',
                     'B15', 'B16', 'B17', 'B18', 'B19']
        bounds = (ind_quads_before_ws24_lb, ind_quads_before_ws24_ub)
        init_field_strengths = self.default_field_strengths_before_ws24      
        self.set_field_strengths(ind_quad_ids_before_ws24, init_field_strengths)
        field_strengths = minimize(scorer, init_field_strengths, var_names, bounds)
        if verbose > 0:
            print '  Desired phases : {:.3f}, {:.3f}'.format(mux, muy)
            print '  Calc phases    : {:.3f}, {:.3f}'.format(*self.get_ref_ws_phases())
            print '  Max betas (Q03 - WS24): {:.3f}, {:.3f}'.format(*self.get_max_betas())
            print '  Betas at target: {:.3f}, {:.3f}'.format(*self.get_betas_at_target())
        
    def get_ref_ws_phases(self):
        """Return x and y phases (mod 2pi) at reference wire-scanner."""
        ws_state = self.trajectory.statesForElement(self.ref_ws_id)[0]
        ws_phases = self.adaptor.computeBetatronPhase(ws_state)
        return ws_phases.getx(), ws_phases.gety()
    
    def set_betas_at_target(self, betas, beta_lim_after_ws24=100., 
                            verbose=0):
        """Vary quads after last wire-scanner to set betas at the target.
        
        Parameters
        ----------
        betas : (beta_x, beta_y)
            The desired beta functions at the target.
        beta_lim_after_ws24 : float
            Maximum beta function to allow between ws24 and the target.
        verbose : int
            If greater than zero, print a before/after summary.
        """
        class MyScorer(Scorer):
            def __init__(self, controller):
                self.controller = controller
                self.betas = betas
                self.beta_lim_after_ws24 = beta_lim_after_ws24
                
            def score(self, trial, variables):
                field_strengths = get_trial_vals(trial, variables)            
                self.controller.set_field_strengths(ind_quad_ids_after_ws24, field_strengths)
                self.controller.track()
                residuals = subtract(self.betas, self.controller.get_betas_at_target())
                cost = norm(residuals)
                return norm(residuals) + self.penalty_function()**2
                
            def penalty_function(self):
                max_betas = self.controller.get_max_betas('RTBT_Diag:WS24', None)
                penalty = 0.
                for max_beta in max_betas:
                    penalty += step_func(max_beta - self.beta_lim_after_ws24)            
                return penalty
            
        scorer = MyScorer(self)
        var_names = ['B26', 'B27', 'B28', 'B29', 'B30']
        init_field_strengths = [self.get_field_strength(quad_id) 
                                for quad_id in ind_quad_ids_after_ws24]     
        bounds = (ind_quads_after_ws24_lb, ind_quads_after_ws24_ub)
        field_strengths = minimize(scorer, init_field_strengths, var_names, bounds)
        self.set_field_strengths(ind_quad_ids_after_ws24, field_strengths)
        if verbose > 0:
            print '  Desired betas: {:.3f}, {:.3f}'.format(*betas)
            print '  Calc betas   : {:.3f}, {:.3f}'.format(*self.get_betas_at_target())
            
    def get_phases_for_scan(self, phase_coverage, npts):
        """Create array of phases for scan. 
        
        Note that this will reset the model optics to their default settings.

        Parameters
        ----------
        phase_coverages : float
            Range of phase advances to cover in radians. The phases are
            centered on the default phase. It is a pain because OpenXAL 
            computes the phases mod 2pi. 
        scans_per_dim : int
            Number of phases to scan for each dimension (x, y). The total 
            number of scans will be 2 * `scans_per_dim`.
        """
        self.restore_default_optics()
        mux0, muy0 = self.get_ref_ws_phases()
        
        def _get_phases_for_scan_1d(dim='x'):
            phase = {'x': mux0, 'y': muy0}[dim]
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

    def restore_default_optics(self):
        """Return quad strengths to their default settings."""
        self.set_field_strengths(ind_quad_ids_before_ws24, self.default_field_strengths_before_ws24)
        self.set_field_strengths(ind_quad_ids_after_ws24, self.default_field_strengths_after_ws24)