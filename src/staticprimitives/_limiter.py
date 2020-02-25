#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains methods to restrict distributions to certain limits

"""

#import warnings as _warnings
import numpy as _np
import scipy.stats as _stats
import itertools as _it
import matplotlib.pyplot as _plt


import namedtensors as _nt
import mechanicalstate as _mechanicalstate
class Limiter():

    def __init__(self,tensornamespace, minimalStdDev=0.1, confidence=0.95):

        self.tns = _nt.TensorNameSpace(tensornamespace)
        self.tns.registerTensor('limits_pos', (('r','g','d'),()))
        self.tns.registerTensor('limits_neg', (('r','g','d'),()))
        
        self.confidenceFactor = 1.0 / _stats.norm.ppf(confidence) #compute the mutiple of sigma that is tolerated

        self.registerTensor('minimalStdDev',  (('r','g','d'),()))
        self.tns.setTensor('minimalStdDev', minimalStdDev)
    
    def setLimits(self, positionlimits, velocitylimits, torquelimits):
    
        self['limits_pos'].data[0,0,:] = positionlimits[0,:]
        self['limits_neg'].data[0,0,:] = positionlimits[1,:]

        self['limits_pos'].data[0,1,:] = velocitylimits[0,:]
        self['limits_neg'].data[0,1,:] = velocitylimits[1,:]

        self['limits_pos'].data[1,0,:] = torquelimits[0,:]
        self['limits_neg'].data[1,0,:] = torquelimits[1,:]

    

    def limit(self, distribution):
        
        #first, limit all means to the interval limits minus the minimal variance:
        limits_neg = self['limits_neg'].data
        limits_pos = self['limits_pos'].data
        means_clipped = _np.clip(distribution.means, self.limits_neg+self.tns['minimalStdDev'].data, self.limits_pos-self.tns['minimalStdDev'].data)

        # compute maximum stddev w.r.t. distances to limits
        stddevs_max_neg = (means_clipped - self.limits_neg) * self.confidenceFactor
        stddevs_max_pos = (self.limits_pos - means_clipped) * self.confidenceFactor
        stddevs_max = _np.fmin(stddevs_max_neg,stddevs_max_pos)

        
        #scale cov matrix so that variances stay mostly within limits:
        stddevs_cov = _np.sqrt(distribution.variances_view)
        scalingfactors  = _np.fmin(stddevs_max /  stddevs_cov , 1.0)
        covmatrix_scaled = distribution.covariances * scalingfactors[:,:,:,_np.newaxis,_np.newaxis,_np.newaxis] * scalingfactors[_np.newaxis,_np.newaxis,_np.newaxis,:,:,:]
        
        #todo: limit velocity so that position is not violated within the safety interval:
        
        return _mechanicalstate.MechanicalStateDistribution(means_clipped, covmatrix_scaled)
        


