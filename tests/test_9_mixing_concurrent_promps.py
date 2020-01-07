#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test mixing of two promps with a varying mixing ratio

The figures should show the confidence area of two MP's which both encode a linear motion
but in opposite direction, and the confidence area of the mixed state distribution when using a
linear "cross-fade"

The first figure shows a cross-fade from MP1 to MP2 (happy mouth)
The second figure shows a constant mix (zero motion)
The third figure shows a cross-fade from MP2 to MP1 (sad mouth)

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import matplotlib.pylab as pylab

#try to change this:
n=10 #nr of supports / basis functions

meansMatrix = _np.linspace(50,-50, n) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)
meansMatrix2 = _np.linspace(-50,50, n) #+ _np.random.normal(0, 5, n)
meansMatrix2.shape = (n, 1)

covarianceTensor = _np.zeros((n,1,n,1))
covarianceTensor2 = _np.zeros((n,1,n,1))
for i,c in enumerate(_np.linspace(0.01,0.99,n)):
    covarianceTensor[i,0,i,0] =  15**2 #10*10**(2*(1-c))
    covarianceTensor2[i,0,i,0] = 15**2 #10*10**(2*(c))


mp1 = promp.ProMP(meansMatrix, covarianceTensor, name='mp1')
mp2 = promp.ProMP(meansMatrix2, covarianceTensor2, name='mp2')

mp1.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True, withGainsPlots=False)
mp2.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True, withGainsPlots=False)

promps = [mp1, mp2]
mixer = promp.ProMPMixer(ProMPList=promps)

#blend from mp1 to mp2 - should give a "happy mouth"
dp=50
activations = [_np.linspace(1.0,0.0,dp)]
activations.append(1.0 - _np.sum(activations, 0))
activations = _np.array(activations)
mixer.plot(activations)

#blend equally - means should cancel out to 0
activations2 = [_np.linspace(0.5,0.5,dp)]
activations2.append(1.0 - _np.sum(activations2, 0))
activations2 = _np.array(activations2)
mixer.plot(activations2)

#blend from mp2 to mp1- should give a "sad mouth"
activations3 = [_np.linspace(0.0,1.0,dp)]
activations3.append(1.0 - _np.sum(activations3, 0))
activations3 = _np.array(activations3)
mixer.plot(activations3,)

meansMatrix3 = _np.linspace(150,150, n) #+ _np.random.normal(0, 5, n)
meansMatrix3.shape = (n, 1)
meansMatrix4 = _np.linspace(50,-50, n) #+ _np.random.normal(0, 5, n)
meansMatrix4.shape = (n, 1)

mp3 = promp.ProMP(meansMatrix3, covarianceTensor, name='mp3')
mp4 = promp.ProMP(meansMatrix4, covarianceTensor, name='mp4')

promps = [mp3, mp4]
mixer = promp.ProMPMixer(ProMPList=promps)
activations3 = [_np.hstack([_np.linspace(0.0,1.0,dp/2),_np.linspace(1.0,0.0,dp/2)])]
#activations3 = [_np.hstack([_np.linspace(0.5,0.5,dp/2),_np.linspace(0.5,0.5,dp/2)])]
activations3.append(1.0 - _np.sum(activations3, 0))
activations3 = _np.array(activations3)
mixer.plot(activations3)

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
