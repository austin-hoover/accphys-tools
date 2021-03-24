import math
import random
from xal.smf import Accelerator
from xal.smf.data import XMLDataManager
from xal.extension.solver import Trial, Variable, Scorer, Stopper, Solver, Problem
from xal.extension.solver.ProblemFactory import getInverseSquareMinimizerProblem
from xal.extension.solver.SolveStopperFactory import maxEvaluationsStopper
from xal.extension.solver.algorithm import SimplexSearchAlgorithm

from mathfuncs import subtract, norm, step_func, dot, put_angle_in_range


# Module level variables
init_twiss = {'ax': -1.378, 'bx': 6.243, 'ay':0.645, 'by':10.354} # RTBT entrance
design_betas_at_target = (57.705, 7.909)


def loadRTBT():
    """Load the RTBT sequence of the SNS accelerator."""
    accelerator = XMLDataManager.acceleratorWithPath(
        '/Users/46h/Research/code/snsxal/site/optics/design/main.xal')
    return accelerator.getComboSequence('RTBT')


def write_traj_to_file(data, positions, filename):
    """Save trajectory data to file. 
    `data[i]` is list of data at position `positions[i]`."""
    file = open(filename, 'w')
    fstr = len(data[0]) * '{} ' + '{}\n'
    for s, dat in zip(positions, data):
        file.write(fstr.format(s, *dat))
    file.close()
    

# Helper functions for OpenXAL Solver
def get_trial_vals(trial, variables):
    """Get list of variable values from Trial."""
    trial_point = trial.getTrialPoint()
    return [trial_point.getValue(var) for var in variables]


def minimize(variables, scorer, solver, tol=1e-8):
    """Run the solver to minimize the score."""
    problem = getInverseSquareMinimizerProblem(variables, scorer, tol)
    solver.solve(problem)
    trial = solver.getScoreBoard().getBestSolution()
    return get_trial_vals(trial, variables)


def solve(scorer, init_vals, var_names, bounds, maxiters=1000, tol=1e-8):
    """Given a Scorer and the, initial conditions, set up the problem and
    minimize the score using the simplex algorithm."""
    lb, ub = bounds
    variables = [Variable(name, val, l, u) for name, val, l, u 
                 in zip(var_names, init_vals, lb, ub)]
    stopper = maxEvaluationsStopper(maxiters)
    solver = Solver(SimplexSearchAlgorithm(), stopper)
    return minimize(variables, scorer, solver, tol)
    

def least_squares(A, b, x0=None, lb=None, ub=None, verbose=0):
    """Return the least-squares solution to the equation A.x = b.
    
    This will be used if we want to reconstruct the beam emittances from 
    within the app.
    """ 
    class MyScorer(Scorer):
        def __init__(self, A, b):
            self.A, self.b = A, b
        def score(self, trial, variables):
            x = get_trial_vals(trial, variables)
            residuals = subtract(dot(A, x), b)
            return norm(residuals)
    
    n = len(A[0])
    var_names = ['v{}'.format(i) for i in range(n)] 
    x0 = [random.random() for _ in range(n)] if x0 is None else x0
    lb = n * [-float('inf')] if lb is None else lb
    ub = n * [+float('inf')] if ub is None else ub
    max_iters = 1000
    tol = 1e-8

    variables = [Variable(name, val, l, u) for name, val, l, u
                 in zip(var_names, x0, lb, ub)]

    scorer = MyScorer(A, b)
    stopper = maxEvaluationsStopper(max_iters)
    solver = Solver(SimplexSearchAlgorithm(), stopper)
    problem = getInverseSquareMinimizerProblem(variables, scorer, tol)
    solver.solve(problem)
    trial = solver.getScoreBoard().getBestSolution()
    x = get_trial_vals(trial, variables)
    if verbose > 0:
        print solver.getScoreBoard()
    return x 