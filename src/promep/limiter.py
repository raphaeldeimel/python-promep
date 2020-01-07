#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains methods to restrict distributions to certain limits

"""
import warnings
import numpy as _np
import scipy.stats as _stats
import itertools as _it
import matplotlib.pyplot as _plt

from . import _tensorfunctions as _t
from . import _timeintegrator

class Limiter():

    def __init__(self,mechanicalStateDescription, minimalStdDev=0.1, confidence=0.95):
        self._md  = mechanicalStateDescription
        self.limits_neg = _np.zeros((self._md.mechanicalStatesCount,self._md.dofs))
        self.limits_pos = _np.zeros((self._md.mechanicalStatesCount,self._md.dofs))
        self.confidenceFactor = 1.0 / _stats.norm.ppf(confidence) #compute the mutiple of sigma that is tolerated
        self.minimalStdDev = _np.full((self._md.mechanicalStatesCount,self._md.dofs),minimalStdDev)
    
    def setLimits(self, positionlimits, velocitylimits, torquelimits):
        self.limits_neg[self._md.mStateNames2Index['position'],:] = positionlimits[0,:]
        self.limits_pos[self._md.mStateNames2Index['position'],:] = positionlimits[1,:]
        
        self.limits_neg[self._md.mStateNames2Index['velocity'],:] = velocitylimits[0,:]
        self.limits_pos[self._md.mStateNames2Index['velocity'],:] = velocitylimits[1,:]

        self.limits_neg[self._md.mStateNames2Index['torque'],:] = torquelimits[0,:]
        self.limits_pos[self._md.mStateNames2Index['torque'],:] = torquelimits[1,:]
    

    def limit(self, distribution):
        mean, covmatrix = distribution
        
        #first, limit all means to the interval limits minus the minimal variance:
        means_clipped = _np.clip(mean, self.limits_neg+self.minimalStdDev, self.limits_pos-self.minimalStdDev)

        # compute maximum stddev w.r.t. distances to limits
        stddevs_max_neg = (means_clipped - self.limits_neg) * self.confidenceFactor
        stddevs_max_pos = (self.limits_pos - means_clipped) * self.confidenceFactor
        stddevs_max = _np.fmin(stddevs_max_neg,stddevs_max_pos)

        
        #scale cov matrix so that variances stay mostly within limits:
        stddevs_cov = _np.sqrt(_t.getDiagView(covmatrix))
        scalingfactors  = _np.fmin(stddevs_max /  stddevs_cov , 1.0)
        covmatrix_scaled = covmatrix * scalingfactors[:,:,_np.newaxis,_np.newaxis] * scalingfactors[_np.newaxis,_np.newaxis,:,:]
        
        #todo: limit velocity so that position is not violated within the safety interval:
        
        return means_clipped, covmatrix_scaled
        


