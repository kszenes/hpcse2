#!/usr/bin/env python
import numpy as np

# define the egg-crate function
def egg_crate(p):
  
  # get the parameters TODO
  x = p["Parameters"][0]
  y = p["Parameters"][1]

  # calculate the result TODO
  res = x**2 + y**2 + 25 * (np.sin(x)**2 + np.sin(y)**2)

  # return the result TODO
  p["F(x)"] = - res
