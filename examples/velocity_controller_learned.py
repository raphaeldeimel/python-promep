#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:19:06 2017

Creates a ProMP to achieve a specific velocity at the end

@author: raphael
"""

import sys
sys.path.insert(0,"../src/")
import promp


import numpy as _np
import matplotlib.pylab as pylab

#try to change this:
nSupports=4 #nr of supports / basis functions
dofs = 1
nSteps= 20
nTrajectories = 100


trajectorypositions = []
phases = _np.linspace(0.0,1.0,nSteps)

dqd  = 10.0
kd=0.2
dp=1.0 / nSteps
for i in range(nTrajectories):
    q =  _np.random.normal(0.0, 10.0, size=dofs) #start pos
    dq = _np.random.normal(-10.0, 10.0, size=dofs)   #start velocity
    trajectory = _np.zeros((dofs, nSteps))
    for i in range(nSteps):
        dq = dq + kd*(dqd - dq)  #controller
        q = q + dq * dp          #phasestep
        trajectory[:,i] = q
    trajectorypositions.append(trajectory)

pylab.figure()
for t in trajectorypositions:
    pylab.plot(phases, t[0,:])

M=0

mp1 = promp.ProMPFactory.makeFromPositionTrajectories("constantvelocity", nSupports, trajectorypositions)

mp1.plot()

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
