#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:19:06 2017

This test creates a ProMP with fake torques


@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab

#try to change this:
n=5 #nr of supports / basis functions

meansMatrix =  10.0 *_np.sin( _np.linspace(-0.2*_np.pi, 2.2*_np.pi, n)) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)

sigmas= _np.diag(10.0 *_np.cos( _np.linspace(0.0, 1*_np.pi, n))) #+ _np.random.normal(0, 5, n)
#sigmas = 10*_np.diag(_np.random.chisquare(4, n))/4
corr_matrix = _np.full((n,n), 0.0)
_np.fill_diagonal(corr_matrix, 1.0)
covarianceTensor = _np.dot(sigmas, _np.dot(corr_matrix,sigmas)) #  = sigmas @ corr_matrix @ sigmas
covarianceTensor.shape = (n,1,n,1)

mp1 = promp.ProMP(meansMatrix, covarianceTensor, fakeSigmaTaus=[30.0, -5.5, -25.25]) #fake covariances for tau-tau, tau-pos, tau-vel 
mp1.plot(withSampledTrajectories=5, withConfidenceInterval=True)

#test setting/removing fake torques after instantiation:
mp2  = promp.ProMP(meansMatrix, covarianceTensor)
mp2.plot(withSampledTrajectories=5, withConfidenceInterval=True)
mp2.fakeEffort( [26.0, -25.1, -0.0] )  #fake covariances for tau-tau, tau-pos, tau-vel 
mp2.plot(withSampledTrajectories=5, withConfidenceInterval=True)
mp2.fakeEffort(None)
mp2.plot(withSampledTrajectories=5, withConfidenceInterval=True)

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
