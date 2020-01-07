#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:19:06 2017

test a deterministic multi-dof trajectory

The figure should show four individual trajectories with different oscillation frequencies

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab

#with more dofs:
dofs=3
supports = 7
meansMatrix = _np.vstack( [ 10*_np.sin(_np.linspace(0, i*_np.pi, supports))  for i in range(dofs)] ).T
#meansMatrix = _np.vstack( [ _np.linspace(0, 10, supports)  for i in range(dofs)] ).T
sigmas = _np.vstack( [_np.linspace(0.1, 0.1, supports)  for i in range(dofs)] ).T

covariances = _np.zeros((supports,dofs,supports,dofs))
for dof in range(dofs):
    for i in range(supports):
        covariances[i,dof,i, dof]  = sigmas[i, dof]**2
        if i+1 < (supports):
            covariances[i,dof, i+1,dof] =  0.0 * sigmas[i, dof] * sigmas[i+1, dof]
            covariances[i+1,dof ,i,dof] = covariances[i,dof, i+1,dof]
        if i+2 < (supports):
            covariances[i,dof, i+2,dof] =  0.0 * sigmas[i, dof] * sigmas[i+2, dof]
            covariances[i+2,dof, i,dof] = covariances[i,dof, i+1,dof]
        if i+3 < (supports):
            covariances[i,dof, i+3,dof] =  0.0 * sigmas[i, dof] * sigmas[i+3, dof]
            covariances[i+3,dof, i,dof] = covariances[i,dof, i+1,dof]



mp2 = promp.ProMP(meansMatrix, covariances)

mp2.plot(withSampledTrajectories=2)
#mp2.plot(dofs=[0,3], withSampledTrajectories=10)
covariancesFlat= promp.flattenCovarianceTensor(covariances)
covariancesActualGroupedBySupport = _np.cov(_np.vstack([mp2.sample().flatten() for i in range(10000)]).T)




if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
