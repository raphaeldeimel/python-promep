#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test mixing of two promps with a varying mixing ratio

This test switches between two concurrent promps several times

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import scipy
import matplotlib.pylab as pylab

#try to change this:
n=10 #nr of supports / basis functions

covarianceTensor = _np.zeros((n,1,n,1))
covarianceTensor2 = _np.zeros((n,1,n,1))
for i,c in enumerate(_np.linspace(0.01,0.99,n)):
    covarianceTensor[i,0,i,0] =  10**2 #10*10**(2*(1-c))
    covarianceTensor2[i,0,i,0] = 100**2 #10*10**(2*(c))


meansMatrix = _np.linspace(50,50, n) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)
meansMatrix2 = _np.linspace(200,200, n) #+ _np.random.normal(0, 5, n)
meansMatrix2.shape = (n, 1)

mp1 = promp.ProMP(meansMatrix, covarianceTensor, name="mp1")
mp2 = promp.ProMP(meansMatrix2, covarianceTensor2, name="mp2")

promps = [mp1, mp2]
mixer = promp.ProMPMixer(ProMPList=promps)
dp=20
#activations3 = [scipy.special.betainc(3,3,_np.hstack([_np.linspace(0.0,1.0,dp/2),_np.linspace(1.0,0.0,dp/2)]))]
activations3 = [_np.hstack([
        _np.linspace(0.0,1.0,dp),
        _np.linspace(1.0,1.0,dp),
        _np.linspace(1.0,0.0,dp),
        _np.linspace(0.0,0.0,dp),
        _np.linspace(0.0,1.0,dp),
        _np.linspace(1.0,1.0,dp),
        _np.linspace(1.0,0.0,dp),
        _np.linspace(0.0,0.0,dp),
        _np.linspace(0.0,1.0,dp),
        _np.linspace(1.0,1.0,dp),
        _np.linspace(1.0,0.0,dp),
        _np.linspace(0.0,0.0,dp),
])]
activations3.append(1.0 - activations3[0])
activations3 = _np.vstack(activations3)
mixer.plot(activations3)
mixer.plot(activations3, sigma_control=100.0)

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
