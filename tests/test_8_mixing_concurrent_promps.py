#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps using a constant mixing ratio (0.5 + 0.5)

The final figure should show the confidence areas of the two initial ProMPs, and the
confidence area of the mixed state distribution. In each phase instant, the mixed state
distribution should tend to follow the MP that is more certain (i.e. which has narrower confidence interval)

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab

#try to change this:
n=11 #nr of supports / basis functions

meansMatrix =  20.0 *_np.sin( _np.linspace(-0.2*_np.pi, 2.2*_np.pi, n)) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)
meansMatrix2 = -40.0 *_np.sin( _np.linspace(-0.2*_np.pi, 2.2*_np.pi, n)) #+ _np.random.normal(0, 5, n)
meansMatrix2.shape = (n, 1)

covarianceTensor = _np.zeros((n,1,n,1))
covarianceTensor2 = _np.zeros((n,1,n,1))
for i,c in enumerate(_np.linspace(0.01,0.99,n)):
    covarianceTensor[i,0,i,0] =  10*10**(2*(1-c))
    covarianceTensor2[i,0,i,0] = 10*10**(2*(c))

mp1 = promp.ProMP(meansMatrix, covarianceTensor, name="mp1")
mp2 = promp.ProMP(meansMatrix2, covarianceTensor2, name="mp2")

mp1.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True,withGainsPlots=False)
mp2.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True,withGainsPlots=False)

promps = [mp1, mp2]
md = mp1._md
mixerP = promp.ProMPMixer(md, ProMPList=promps, weighingMethod='Paraschos')
mixerD = promp.ProMPMixer(md, ProMPList=promps, weighingMethod='Deimel')

activationVector= [0.5,0.5]
phase = [0.2, 0.2]
phaseVelocity  = [1.0, 1.0]
activations = [_np.linspace(0.5,0.5,100)]
activations.append(1.0 - _np.sum(activations, 0))
activations = _np.array(activations)

ylimits = {'position': (-50,50), 'velocity': (-300,300)},

means, sigmas, third = mixerP.getMixedDistribution(activationVector, phase, phaseVelocity)
mixerP.plot(activations, withGainsPlots=False, ylimits=ylimits)
means, sigmas, third = mixerD.getMixedDistribution(activationVector, phase, phaseVelocity)
mixerD.plot(activations, withGainsPlots=False, ylimits=ylimits)

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
