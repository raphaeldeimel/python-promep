#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp
import promp._kumaraswamy

import numpy as _np
import itertools as _it
import scipy
import matplotlib
import matplotlib.pylab as pylab

#try to change this:
nSupports=9 #nr of supports / basis functions
dofs = 1
nSteps= 100
nTrajectories = 10
expectedDuration=3.0
kp=50
kv=5

positionsList = []
velocitiesList = []
torquesList = []
p = promp._kumaraswamy.cdf(1.645,1.800, _np.linspace(0.0,1.0,nSteps+1))
phaseVelocities =  nSteps/expectedDuration * (p[1:] - p[:-1])
phases          = 0.5*(p[1:] + p[:-1])

for i in range(nTrajectories):
    a = _np.zeros((dofs, nSteps+1))
    
    a_error =  _np.random.normal(0.0, 1.0) * _np.linspace(1.0,0.3,nSteps+1)[None, :]
    da_error = _np.random.normal(0.0, 1.0 / expectedDuration ) * phaseVelocities[None, :]
    a[0,:] = p * 10.0  + a_error
    
    pos = 0.5* (a[:,:-1] + a[:,1:])
    vel = (nSteps/expectedDuration) * ( a[:,1:] - a[:,:-1]) + da_error
    tau = -(kp*phaseVelocities*expectedDuration) * 0.5* (a_error[:,:-1] + a_error[:,1:]) - kv * da_error
    positionsList.append(pos)
    velocitiesList.append(vel)
    torquesList.append(tau)


pylab.subplot(2,1,1)
time = _np.linspace(0, expectedDuration,nSteps)
for pos in positionsList:
    pylab.plot(time, pos[0,:])
pylab.subplot(2,1,2)
for pos in positionsList:
    pylab.plot(phases, pos[0,:])

zerotorqueslist = [ 0.0 * tau for tau in torquesList]
mp1  = promp.ProMPFactory.makeFromTrajectories(
        "learning_test", 
        nSupports, 
        positionsList, 
        velocitiesList,
        zerotorqueslist, 
        phasesList=phases, 
        phaseVelocitiesList=phaseVelocities,
        initialVariance=1e-3,
        computeMeanFromSubset=None,
        expectedDuration = expectedDuration,
        useSyntheticTorques=False,
        phaseVelocityRelativeFloor=0.1,
)

mp1.plot()


mp2  = promp.ProMPFactory.makeFromTrajectories(
        "learning_test", 
        nSupports, 
        positionsList, 
        velocitiesList,
        torquesList, 
        phasesList=phases, 
        phaseVelocitiesList=phaseVelocities,
        initialVariance=1e-3,
        computeMeanFromSubset=None,
        expectedDuration = expectedDuration,
        useSyntheticTorques=False,
        phaseVelocityRelativeFloor=0.1,
)

mp2.plot()


mp3  = promp.ProMPFactory.makeFromTrajectories(
        "learning_test", 
        nSupports, 
        positionsList, 
        velocitiesList,
        torquesList, 
        phasesList=phases, 
        phaseVelocitiesList=phaseVelocities,
        initialVariance=1e-3,
        computeMeanFromSubset=None,
        expectedDuration = 10.0,
        useSyntheticTorques=True,
        syntheticKp=20,
        syntheticKv=5,
        phaseVelocityRelativeFloor=0.1,
)
mp3.plot()

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
