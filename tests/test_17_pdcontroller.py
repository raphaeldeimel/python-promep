#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../src/")
import promp


import numpy as _np
import matplotlib.pylab as pylab



#try to change this:
dofs = 1
nTrajectories = 10

#mechanical state description:
md = promp.MechanicalStateDistributionDescription(dofs=dofs)
mstates = md.mechanicalStatesCount
mStateNames = md.mStateNames
iPos = md.mStateNames2Index['position']
iVel = md.mStateNames2Index['velocity']
iTau = md.mStateNames2Index['torque']


# controller parameters, for a specific timestep
kv =_np.array([0])
kp = _np.array([100])
virtualDamping = _np.array([10]) #simulate damping of the system
massMatrixInverse = _np.eye(1) / 0.4567
dt = 0.010

c = promp.PDController(kp, kv, kd=virtualDamping, mstateDescription=md)

mean = _np.zeros((mstates,dofs))
mean[iPos, 0] = 5.0
mean[iVel, 0] = 20.0
cov = _np.zeros((mstates,dofs, mstates,dofs))
dist_initial = (mean, cov)
cov[iVel,0,iVel,0] = 0.5**2
cov[iPos,0,iPos,0] = 5.0**2


numIter= int(1.0 / dt)
t = _np.zeros(numIter)
sigmas = _np.zeros((mstates, numIter))
means = _np.zeros((mstates, numIter))
dist = dist_initial

timeintegrator = promp.TimeIntegrator(dofs)

for i in range(numIter):
    dof=0
    dist = timeintegrator.integrate(dist, dt)
    dist = c.getInstantStateVectorDistribution(currentDistribution=dist)
    t[i] = t[i-1] + dt
    means[:,i] = dist[0][:,dof]
    var = _np.diag(dist[1][:,dof,:,dof])
    sigmas[:,i] = _np.sqrt(var)

pylab.figure()    
pylab.plot(t, means[iTau,:])
pylab.plot(t, means[iTau,:]+sigmas[iTau,:])
pylab.plot(t, means[iTau,:]-sigmas[iTau,:])
pylab.title("torque (mean, +-stddev)")

pylab.figure()    
pylab.plot(t, means[iPos,:])
pylab.plot(t, means[iPos,:]+sigmas[iPos,:])
pylab.plot(t, means[iPos,:]-sigmas[iPos,:])
pylab.title("position (mean, +-stddev)")
    
pylab.figure()    
a = 1
pylab.plot(t, means[iVel,:]*a)
pylab.plot(t, (means[iVel,:]+sigmas[iVel,:])*a)
pylab.plot(t, (means[iVel,:]-sigmas[iVel,:])*a)
pylab.title("velocity (mean, +-stddev)")

velFromPos = (means[iPos,1:]-means[iPos,:-1]) / dt
pylab.plot(0.5*t[1:]+0.5*t[:-1], velFromPos, linestyle=":", linewidth=2.0)

c.plot(num=100, distInitial=dist_initial)

c = promp.PDController(kp, kv, mstateDescription=md)  #undamped controller
c.plot(num=100, distInitial=dist_initial)

c = promp.PDController(0*kp, 20.0+kv, mstateDescription=md)  #kv-damped controller:
c.plot(num=100, distInitial=dist_initial)


c = promp.PDController(kp, kv, kd=virtualDamping, desiredPosition=100.0, mstateDescription=md)
mean = _np.zeros((mstates,1))
cov = _np.zeros((mstates,1, mstates,1))
cov[iPos,0,iPos,0] = 2.0**2
cov[iVel,0,iVel,0] = 2.0**2
dist_initial = (mean, cov)
c.plot(num=100, duration=1.0, distInitial=dist_initial)
pylab.suptitle("zero desired position")

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
