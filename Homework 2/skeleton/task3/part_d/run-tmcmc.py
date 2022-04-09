#!/usr/bin/env python3

# In this example, we demonstrate how Korali samples the posterior distribution
# in a bayesian problem where the likelihood is calculated by providing
# reference data points and their objective values.


# Importing the computational model
import sys
sys.path.append('./_model')
from model import *

# Import Korali TODO
import korali

# Create Korali engine TODO
k = korali.Engine()

# Creating new experiment TODO
e = korali.Experiment()

# Setting up the reference likelihood for the Bayesian Problem TODO
e["Problem"]["Type"] = "Bayesian/Reference"
e["Problem"]["Likelihood Model"] = "Normal"
e["Problem"]["Reference Data"] = getReferenceData()
e["Problem"]["Computational Model"] = lambda sampleData: model(sampleData, getReferencePoints())

# Configuring TMCMC parameters
e["Solver"]["Type"] = "Sampler/TMCMC"
e["Solver"]["Population Size"] = 5000
e["Solver"]["Target Coefficient Of Variation"] = 0.5
e["Solver"]["Covariance Scaling"] = 0.04


# Configuring the problem's random distributions TODO
e["Distributions"][0]["Name"] = "Uniform 0"
e["Distributions"][0]["Type"] = "Univariate/Uniform"
e["Distributions"][0]["Minimum"] = -5.0
e["Distributions"][0]["Maximum"] = +5.0

e["Distributions"][1]["Name"] = "Uniform 1"
e["Distributions"][1]["Type"] = "Univariate/Uniform"
e["Distributions"][1]["Minimum"] = 0.0
e["Distributions"][1]["Maximum"] = +5.0

# Configuring the problem's variables and their prior distributions TODO
e["Variables"][0]["Name"] = "c"
e["Variables"][0]["Prior Distribution"] = "Uniform 0"

e["Variables"][1]["Name"] = "k1"
e["Variables"][1]["Prior Distribution"] = "Uniform 0"

e["Variables"][2]["Name"] = "k3"
e["Variables"][2]["Prior Distribution"] = "Uniform 0"

e["Variables"][3]["Name"] = "k5"
e["Variables"][3]["Prior Distribution"] = "Uniform 0"

e["Variables"][4]["Name"] = "[Sigma]"
e["Variables"][4]["Prior Distribution"] = "Uniform 1"




e["Store Sample Information"] = True


# Configuring output settings TODO


# Set console Verbosity
e["Console Output"]["Verbosity"] = "Detailed"

# Run Experiment TODO
k.run(e)
