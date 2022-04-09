#!/usr/bin/env python
import numpy as np

# 1-d problem
def model_1d(p):
  x = p["Parameters"][0]
  p["F(x)"] = -0.5 * x * x
