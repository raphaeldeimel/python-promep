#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import itertools as _it
import scipy
import matplotlib.pylab as pylab

#try to change this:
n=7 #nr of supports / basis functions

#c = _np.array([0.2,-0.25,0.0,0.5,1.0,1.25,0.8])
c = _np.array([-0.25,0.0,0.25,0.5,0.75,1.0,1.25])

meansMatrix = 0 + 50 * c # _np.linspace(50,50, n) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)
meansMatrix2 = 50 - 200 * c # _np.linspace(85,-185, n) #+ _np.random.normal(0, 5, n)
meansMatrix2.shape = (n, 1)
meansMatrix3 = -150 + 150 * c #_np.linspace(-185,85, n) #+ _np.random.normal(0, 5, n)
meansMatrix3.shape = (n, 1)

#covarianceTensor = _np.zeros((n,1,1,n))
#for i,c in enumerate(_np.linspace(0.01,0.99,n)):
#    covarianceTensor[i,0,0,i] =  1000*2**(-3-30*c*(1-c))

sigma=5
covarianceTensor = promp.makeCovarianceTensorUncorrelated(n, 1, sigma)


mp1 = promp.ProMP(meansMatrix, covarianceTensor, fakeSigmaTaus=[0.4, 0.2,0.2], name='mp12')
mp2 = promp.ProMP(meansMatrix2, covarianceTensor, fakeSigmaTaus=[0.4, 0.2,0.2], name='mp23')
mp3 = promp.ProMP(meansMatrix3, covarianceTensor, fakeSigmaTaus=[0.4, 0.2,0.2], name='mp31')


mp11 =  promp.PDController([5], [5],  desiredPosition = [0], name='pd1')
mp22 =  promp.PDController([5], [5],  desiredPosition = [50], name='pd2')
mp33 =  promp.PDController([5], [5],  desiredPosition = [-150], name='pd3')



mp1.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True, withGainsPlots=False)
mp2.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True, withGainsPlots=False)
mp3.plot(withSampledTrajectories=5, withSupportsMarked=False, withConfidenceInterval=True, withGainsPlots=False)

promps = [mp1, mp2, mp3]
stateController = [mp11, mp22, mp33]
transitionsList = [(1,0),(2,1), (0,2)]
mixer = promp.ProMPMatrixMixer(mp1._md, promps, transitionsList, stateControllerList=stateController, weighingMethod='Deimel')

#blend from mp1 to mp2 - should give a "happy mouth"
activationsMatrix = _np.zeros((3,3))
phasesMatrix = _np.zeros((3,3))

#fake phase state machine:
#n1, n2, n3 = 70,10,5  # duration of transition, duration of mixing transition-state and dureation of state
n1, n2, n3 = 50,20,1  # duration of transition, duration of mixing transition-state and dureation of state
nsum=n1 + 2*n2 + 2*n3
activationState = _np.hstack([
        _np.linspace(1,1, n3), _np.linspace(1,0, n2),
        _np.linspace(0,0, n1+n2+n3),
        _np.linspace(0,0, nsum),
        _np.linspace(0,0, n1+n2+n3),
        _np.linspace(0,1, n2),
        _np.linspace(1,1, n3),
])
activation11 = _np.roll(activationState, 0)
activation22 = _np.roll(activationState, nsum)
activation33 = _np.roll(activationState, 2*nsum)

activation12 = _np.hstack([
        _np.linspace(0,0, n3),
        _np.linspace(0,1, n2),
        _np.linspace(1,1, n1),_np.linspace(1,0, n2),
        _np.linspace(0,0, nsum+n3),
        _np.linspace(0,0, nsum),
])

activation23 = _np.roll(activation12, nsum)
activation31 = _np.roll(activation12, 2*nsum)


lo = 0.0
hi=1.0
phase12 = _np.hstack([
        _np.linspace(lo,lo, n3),
        _np.linspace(lo,hi, nsum-2*n3),
        _np.linspace(hi,hi, 2*n3),
        _np.linspace(hi,lo, nsum-2*n3),
        _np.linspace(lo,lo, nsum+n3),
])

phase23 = _np.roll(phase12, nsum)
phase31 = _np.roll(phase12, 2*nsum)



pylab.figure("activations")
pylab.plot(activation11);pylab.plot(activation12);pylab.plot(activation22);
pylab.plot(activation23);pylab.plot(activation33);pylab.plot(activation31);
pylab.figure("phases")
pylab.plot(phase12);pylab.plot(phase23);pylab.plot(phase31);

means = _np.empty((activation12.size))
sigmas = _np.empty((activation12.size))

phaseState = _np.full(phase12.shape,0.5)

activationsAll = _np.zeros((activation12.size, 3, 3))
activationsAll[:,0,0]= activation11
activationsAll[:,1,1]= activation22
activationsAll[:,2,2]= activation33
activationsAll[:,1,0]= activation12
activationsAll[:,2,1]= activation23
activationsAll[:,0,2]= activation31

phasesAll = _np.zeros((activation12.size, 3, 3))
phasesAll[:,1,0] = phase12
phasesAll[:,2,1] = phase23
phasesAll[:,0,2] = phase31

mixer.plot(activationsAll, phasesAll)

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
