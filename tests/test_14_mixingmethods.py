#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp
import promp._tensorfunctions as _t

import numpy as _np
import itertools as _it
import scipy
import matplotlib.pylab as pylab

#try to change this:
n=7 #nr of supports / basis functions

#c = _np.array([0.2,-0.25,0.0,0.5,1.0,1.25,0.8])
c = _np.array([-0.20,0.1,0.25,0.5,0.75,0.9,1.20])

meansMatrix = 0 + 50 * c # _np.linspace(50,50, n) #+ _np.random.normal(0, 5, n)
meansMatrix.shape = (n, 1)
meansMatrix2 = 50 + 100 * c # _np.linspace(85,-185, n) #+ _np.random.normal(0, 5, n)
meansMatrix2.shape = (n, 1)
meansMatrix3 = 0 * c #_np.linspace(-185,85, n) #+ _np.random.normal(0, 5, n)
meansMatrix3.shape = (n, 1)

#covarianceTensor = _np.zeros((n,1,n,1))
#for i,c in enumerate(_np.linspace(0.01,0.99,n)):
#    covarianceTensor[i,0,i,0] =  1000*2**(-3-30*c*(1-c))

sigma1=10
sigma2=1
sigma3=200


mp1 = promp.ProMP(meansMatrix, _t.makeCovarianceTensorUncorrelated(n, 1, sigma1), fakeSigmaTaus=[0.4, -0.2, -0.2], name='mp12')
mp2 = promp.ProMP(meansMatrix2, _t.makeCovarianceTensorUncorrelated(n, 1, sigma2), fakeSigmaTaus=[1.2, -1.0, -0.2], name='mp23')
mp3 = promp.ProMP(meansMatrix3, _t.makeCovarianceTensorUncorrelated(n, 1, sigma3), fakeSigmaTaus=[0.4, -0.2, -0.2], name='mp31')


mp11 =  promp.PDController([0.5], [1], name='pd11')
mp11.setDesiredPosition([0])
mp22 =  promp.PDController([5], [1], name='pd22')
mp22 .setDesiredPosition([50])
mp33 =  promp.PDController([0.5], [1], name='pd33')
mp33 .setDesiredPosition([150])


#blend from mp1 to mp2 - should give a "happy mouth"
activationsMatrix = _np.zeros((3,3))
phasesMatrix = _np.zeros((3,3))

#fake phase state machine:
n1, n2, n3 = 160,20,40  # duration of transition, duration of mixing transition-state and dureation of state
#n1, n2, n3 = 50,20,1  # duration of transition, duration of mixing transition-state and dureation of state
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


activationsAll = _np.zeros((activation12.size, 3, 3))
activationsAll[:,0,0]= activation11
activationsAll[:,1,1]= activation22
activationsAll[:,2,2]= activation33
activationsAll[:,1,0]= activation12
activationsAll[:,2,1]= activation23
activationsAll[:,2,0]= activation31

phasesAll = _np.zeros((activation12.size, 3, 3))
phasesAll[:,0,0] = 0
phasesAll[:,1,1] = 0
phasesAll[:,2,2] = 0
phasesAll[:,1,0] = phase12
phasesAll[:,2,1] = phase23
phasesAll[:,0,2] = phase31


promps = [mp1, mp2, mp3]
stateControllers = [mp11, mp22, mp33]

ylimits  =[(-5,5),(-50,250), (-10,10) ]
transitionsList = [(1,0),(2,1), (0,2)]

md = mp1._md
mixerP = promp.ProMPMatrixMixer(md, promps, transitionsList, stateControllers, weighingMethod='Paraschos')
mixerP.plot(activationsAll, phasesAll, duration=100.0, ylimits=ylimits)
pylab.suptitle("Interpolation method: Paraschos")

mixerD = promp.ProMPMatrixMixer(md, promps, transitionsList, stateControllers, weighingMethod='Deimel')
mixerD.plot(activationsAll, phasesAll, duration=100.0, ylimits=ylimits)
pylab.suptitle("Interpolation method: Deimel")

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
