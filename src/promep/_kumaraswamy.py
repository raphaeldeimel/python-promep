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
    p = (a*b)*xa1*(1-xa)**(1-b)
    for j in range(x.shape[1]):
        for i in range(x.shape[0]):
            if x[i,j] < 0.0:
                p[i,j] = 0.0
            if  x[i,j] > 1.0:
                p[i,j] = 1.0
    return p
