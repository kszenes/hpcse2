#!/usr/bin/env python3

# In this example, we demonstrate how Korali find the variable values
# that maximize the posterior in a bayesian problem where the likelihood
# is calculated by providing reference data points and their objective values.


# Importing the computational model
import sys
sys.path.append('./_model')
from model import *

# Import Korali TODO
import korali

# Create the engine TODO
k = korali.Engine()

# Creating new experiment TODO
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem #TODO
e["Random Seed"] = 0xC0FEE
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()
e["Problem"]["Computational Model"] = lambda sampleData: model(sampleData, getReferencePoints())


# Configuring the problem's random distributions #TODO
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -5.0
e["Distributions"][0]["Maximum"] = +5.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = -5.0
e["Distributions"][1]["Maximum"] = +5.0

e["Distributions"][2]["Name"] = "Uniform 2"
e["Distributions"][2]["Type"] = "Univariate/Uniform"
e["Distributions"][2]["Minimum"] = -5.0
e["Distributions"][2]["Maximum"] = +5.0

e["Distributions"][3]["Name"] = "Uniform 3"
e["Distributions"][3]["Type"] = "Univariate/Uniform"
e["Distributions"][3]["Minimum"] = 0.0
e["Distributions"][3]["Maximum"] = +5.0

# Configuring the problem's variables #TODO
e["Variables"][0]["Name"] = "k1"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"
e["Variables"][0]["Initial Value"] = +0.0
e["Variables"][0]["Initial Standard Deviation"] = +1.0

e["Variables"][1]["Name"] = "k3"
e["Variables"][1]["Prior Distribution"] = "Uniform 1"
e["Variables"][1]["Initial Value"] = +0.0
e["Variables"][1]["Initial Standard Deviation"] = +1.0

e["Variables"][2]["Name"] = "k5"
e["Variables"][2]["Prior Distribution"] = "Uniform 2"
e["Variables"][2]["Initial Value"] = +0.0
e["Variables"][2]["Initial Standard Deviation"] = +1.0

e["Variables"][3]["Name"] = "[Sigma]"
e["Variables"][3]["Prior Distribution"] = "Uniform 3"
e["Variables"][3]["Initial Value"] = +2.5
e["Variables"][3]["Initial Standard Deviation"] = +0.5

# Configuring CMA-ES parameters
e["Solver"]["Type"] = "Optimizer/CMAES"
e["Solver"]["Population Size"] = 32
e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-14
e["Solver"]["Termination Criteria"]["Max Generations"] = 500


# Configuring output settings #TODO
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_cmaes'
e["File Output"]["Frequency"] = 5
e["Console Output"]["Frequency"] = 1


# Run the experiment #TODO
# k = korali.Engine()
k.run(e)

# Result
# [Korali] Optimum found at:
#          k1 = +1.003e+00
#          k3 = +4.283e-01
#          k5 = -1.351e-02
#          [Sigma] = +4.554e-01
# [Korali] Optimum found: -4.645723e+01