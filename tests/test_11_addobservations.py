#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test adding observations 

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab
import itertools

#with more dofs:
dofs=3
supports = 5
meansMatrix = _np.vstack( [ 10*_np.sin(_np.linspace(0, i*0.5*_np.pi, supports))  for i in range(dofs)] ).T
#meansMatrix = _np.vstack( [ _np.linspace(0, 10, supports)  for i in range(dofs)] ).T
sigmas = _np.vstack( [_np.linspace(2, 5, supports)  for i in range(dofs)] ).T

covariances = _np.zeros((supports,dofs,supports,dofs))
for dof, dof_b in itertools.product(range(dofs),range(dofs)):
    covMatrix = _np.zeros((supports,supports))
    sigmasDof = sigmas[:, dof:dof+1]
    sigmasDof_b = sigmas[:, dof_b:dof_b+1]
    if dof_b == dof:
        corr = 1.0
    else:
        corr = 0.8 
    for i in range(supports):        
        covMatrix[i,i]  = corr * sigmasDof[i]*sigmasDof_b[i]
    covariances[:,dof,:,dof_b] = covMatrix



mp2 = promp.ProMP(meansMatrix, covariances)

phaseObserved=0.5
meansExpected, covariancesExpected= mp2.getInstantStateVectorDistribution(phaseObserved)

meansObserved = meansExpected
sigmasObserved = _np.array([[3,3,3],[10,10,10]])


mp2.plot(withSampledTrajectories=10, withGainsPlots=False)
mp2.conditionToObservation(phaseObserved, meansObserved,sigmasObserved)

cov = mp2.getCovarianceMatrix()

mp2.plot(withSampledTrajectories=10, withGainsPlots=False)
#mp2.plot(dofs=[0,3], withSampledTrajectories=10)
from promp._promp import *

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
