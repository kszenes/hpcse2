#!/usr/bin/env python3

## In this example, we demonstrate how Korali finds values for the
## variables that maximize the objective function, given by a
## user-provided computational model.

# Importing computational model
import sys
import math

# from task2._model.model import egg_crate
sys.path.append('./_model')
from model import *

# Import Korali TODO
import korali

# Starting Korali's Engine TODO
k = korali.Engine()

# Creating new experiment TODO
e = korali.Experiment()

# Configuring Problem
e["Random Seed"] = 0xBEEF
e["Problem"]["Type"] = "Optimization"
e["Problem"]["Objective Function"] = egg_crate


# Defining the problem's variables. TODO
e["Variables"][0]["Name"] = "x"
e["Variables"][0]["Lower Bound"] = -5.0
e["Variables"][0]["Upper Bound"] = +5.0

e["Variables"][1]["Name"] = "x"
e["Variables"][1]["Lower Bound"] = -5.0
e["Variables"][1]["Upper Bound"] = +5.0

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-14
e["Solver"]["Termination Criteria"]["Max Generations"] = 100

# Configuring results path TODO
# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 1
# e["Console Output"]["Verbosity"] = "Silent"

# Result 
# [Korali] Optimum found at:
#         x = -4.219e-09
#         x = +2.051e-09
# [Korali] Optimum found: -5.721571e-16

# Running Korali TODO
k.run(e)
