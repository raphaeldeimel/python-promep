#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:19:06 2017

This test creates a one-dimensional ProMP with 10 supports

The test should create a sinus trajectory with low variance in the middle and
high variance at start and end of the trajectory

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab

#try to change this:
n=10 #nr of supports / basis functions

meansMatrix =  10.0 *_np.sin( _np.linspace(-0.2*_np.pi, 2.2*_np.pi, n)) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)

sigmas= _np.diag(10.0 *_np.cos( _np.linspace(0.0, 1*_np.pi, n))) #+ _np.random.normal(0, 5, n)
#sigmas = 10*_np.diag(_np.random.chisquare(4, n))/4
corr_matrix = _np.full((n,n), 0.0)
_np.fill_diagonal(corr_matrix, 1.0)
covarianceMatrix = _np.dot(sigmas, _np.dot(corr_matrix,sigmas)) #  = sigmas @ corr_matrix @ sigmas
covarianceMatrix.shape = (n,1,n,1)
cov_flat = covarianceMatrix.reshape((n,n))
md = promp.MechanicalStateDistributionDescription(dofs=1, derivativesCountEffort=0)
ik = promp.InterpolationKernelGaussian(count=n, mstateDescription=md)
mp1 = promp.ProMP(meansMatrix, covarianceMatrix, interpolationKernel=ik)
#mp1 = promp.ProMP(meansMatrix, covarianceMatrix)

mp1.plot(withSampledTrajectories=5,withGainsPlots=False)


if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
