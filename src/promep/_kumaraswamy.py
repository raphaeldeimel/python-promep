#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence

This file provides methods for computing the Kumaraswamy distribution's CDF 

This distribution is similar to the beta distribution, but much easier to compute.

"""
import numpy as _np

#try to use numba if installed:
try: 
    from numba import jit
except ImportError: 
    # make a no-op decorator:
    class jit(object):
        def __init__(self, nopython=False):
            pass
        def __call__(self, nopython=False):
            pass


@jit(nopython=True)
def cdf(a, b, x):
    """
    Cumulative density function of the Kumaraswamy distribution
    
    a,b: parameters of the distribution
    
    """
    y = 1.0-(1.0-x**a)**b
    for i in range(x.size):
            if x.flat[i] < 0.0:
                y.flat[i] = 0.0
            if  x.flat[i] > 1.0:
                y.flat[i] = 1.0    
    return y
    
@jit(nopython=True) 
def pdf(a, b, x):
    """
    Probabilitiy density function of the Kumaraswamy distribution
    
    a,b: parameters of the distribution
    
    """
    xa1 = x**(a-1)
    xa = x * xa1
    p = (a*b)*xa1*(1-xa)**(b-1)
    for i in range(x.size):
            if x.flat[i] < 0.0:
                p.flat[i] = 0.0
            if  x.flat[i] > 1.0:
                p.flat[i] = 1.0    
    return p
    
    
    

def approximate(cdf_target, accuracy=2e-3, max_iterations=1000):
    """
    Approximate a given cumulative histogram with a Kumaraswamy distribution
    
    Uses stochastic gradient descent plus a linear cost weighting scheme to stabilize convergence
    """

    n = cdf_target.shape[0]
    x = cdf_target[:,0]
    y = cdf_target[:,1]

    a_ =  1.0
    b_ =  1.0
    costs_ = _np.inf
    error_ = _np.zeros(n)
    epsilon_ = 0.1
    import numpy.random
    for i in range(max_iterations):
        step_a_ = numpy.random.normal(scale=epsilon_) 
        step_b_ = numpy.random.normal(scale=epsilon_) 
        a_proposed = a_ + step_a_
        b_proposed = _np.clip(b_ + step_b_, 0.0, _np.inf)
        cdf_proposed = cdf(a_proposed, b_proposed, x)
        error = (cdf_proposed - y)**2
        costs_proposed = _np.sqrt(_np.sum( error ))/n
        if costs_proposed < costs_:
            mass = (step_a_, step_b_)
            a_ = a_proposed
            b_ = b_proposed
            costs_ = costs_proposed
            error_ = error
            epsilon_ = 10*costs_
        if costs_ < accuracy:
            break
    return  a_, b_, costs_, i
        
