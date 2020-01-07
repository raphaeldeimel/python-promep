#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test a trajectory highly correlated across its two dofs

The figure should show trajectories whose variances for each support are highly correlated

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib
import matplotlib.pylab as pylab
import itertools

#with more dofs:
dofs=2
supports = 9
meansMatrix = _np.zeros((supports,dofs))
#meansMatrix = _np.vstack( [ 10*_np.sin(_np.linspace(0, i*0.5*_np.pi, supports))  for i in range(dofs)] ).T

#meansMatrix = _np.vstack( [ _np.linspace(0, 10, supports)  for i in range(dofs)] ).T
sigmas = _np.vstack( [_np.linspace(2, 10, supports)  for i in range(dofs)] ).T

covariances = _np.zeros((supports,dofs,supports,dofs))
for dof, dof_b in itertools.product(range(dofs),range(dofs)):
    if dof_b != dof:
        corr = 0.999  #correlation between different dofs at the same support
    else:
        corr = 1.0  #correlation between itself
    for i in range(supports):
        j = i  #only between supports at the same phase
        covariances[i,dof,j,dof_b] = corr * sigmas[i, dof] * sigmas[j, dof_b]



mp2 = promp.ProMP(meansMatrix, covariances)

cov = mp2.getCovarianceMatrix()

sampleCount = 7
colorcycle = matplotlib.cycler('color', [matplotlib.cm.spectral(i) for i in _np.linspace(0, 1, sampleCount)])
mp2.plot(withSampledTrajectories=sampleCount, sampledTrajectoryStyleCycler=colorcycle)
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
