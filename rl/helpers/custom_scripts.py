import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

def random_distribution_gen(var_in):
    """
    params_in: var of processing time
    returns: some var that lies within a normal distribution arround the param_in
    """
    if var_in ==0:
        return 0
    else:
        lower = round(0.5*var_in)
        scale = 1 if var_in ==6 else 5 if var_in ==30 else 10 if var_in == 60 else 5
        x = np.arange(-lower, lower)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale = scale) - ss.norm.cdf(xL, scale = scale)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1
        return np.random.choice(x, p = prob) + var_in