# import numpy as np
import math

def P(n):
    return 1. - math.factorial(365)/(365**n * math.factorial(365-n))

def exp_P(n):
    p_n = P(n)
    return n * (10 * p_n - 5 * (1-p_n))

n_list = [(10, 14), (25, 39), (40, 49)]
result = []

for i in n_list:
    n_min, n_max = i
    exp_n = int((n_max + n_min) / 2)
    print(exp_n)
    result.append(exp_P(exp_n))

print(result)
