#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test an mp that gets increasingly certain

The figure should show a set of trajectories that converge to a single motion at
the end of the phase

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab

#try to change this:
n=10 #nr of supports / basis functions

meansMatrix =  20.0 *_np.sin( _np.linspace(-0.2*_np.pi, 2.2*_np.pi, n)) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)

covarianceTensor = _np.zeros((n,1,n,1))
for i in range(n):
    covarianceTensor[i,0,i,0] = 50*2**(6-18*(i/n))


mp1 = promp.ProMP(meansMatrix, covarianceTensor)

mp1.plot(withSampledTrajectories=33, withSupportsMarked=False, withConfidenceInterval=False,withGainsPlots=False)



if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
