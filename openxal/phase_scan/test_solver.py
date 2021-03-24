"""
This script tests the OpenXAL solver. The goal is the minimize the function 
f(x, y) = x**2 + y**2 with the constraint f(x, y) > 1.
"""
import math
from xal.extension.solver import (
    Trial, TrialVeto, Variable, Scorer, Stopper, Solver, 
    SolveStopperFactory, ProblemFactory, Problem)
from xal.extension.solver.algorithm import SimplexSearchAlgorithm, RandomSearch
from xal.extension.solver.constraint import Constraint


def get_trial_vals(trial, variables):
    trial_point = trial.getTrialPoint()
    return [trial_point.getValue(var) for var in variables]


class MyScorer(Scorer):

    def __init__(self, targets):
        self.targets = targets

    def get_residuals(self, trial, variables):
        trial_vals = get_trial_vals(trial, variables)
        return [v - t for v, t in zip(trial_vals, self.targets)]
    
    def score(self, trial, variables):
        trial_vals = get_trial_vals(trial, variables)
        residuals = [v - t for v, t in zip(trial_vals, self.targets)]
        cost = sum([res**2 for res in residuals])
        cost += self.constraint_penalty(trial_vals)
        return cost
        
    def constraint_penalty(self, trial_vals):
        x, y = trial_vals
        diff = x**2 + y**2 - 1
        return 1e3 * math.exp(diff) if diff < 0 else 0
    
    
var_names = ['x', 'y']    
guesses = [-5.0, 5.0]
targets = [0.0, 0.0]
lbounds = [-10, -10]
ubounds = [+10, +10]
max_iters = 1000
tol = 1e-8

variables = [Variable(name, guess, lb, ub) for name, guess, lb, ub 
             in zip(var_names, guesses, lbounds, ubounds)]

scorer = MyScorer(targets)
stopper = SolveStopperFactory.maxEvaluationsStopper(max_iters)
solver = Solver(SimplexSearchAlgorithm(), stopper)
problem = ProblemFactory.getInverseSquareMinimizerProblem(variables, scorer, tol)
solver.solve(problem)
print solver.getScoreBoard()