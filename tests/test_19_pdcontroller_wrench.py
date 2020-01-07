#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../src/")
import promp

import pandadynamicsmodel


import numpy as _np
import matplotlib.pylab as pylab



#try to change this:
dofs = 8

# controller parameters, for a specific timestep
kv =_np.array([0,0,0,0,0,0,0,0])
kp = _np.array([0,0,0,0,0,0,0,0])
massMatrixInverse = _np.eye(1) / 0.4567
dt = 0.010


c = promp.PDController(kp=kp,kv=kv,dofs=dofs,desiredEEWrench = [0,0,-1,0,0,0])
mstates = c.mechanicalStatesCount
mStateNames = c.mStateNames
iPos = c._iPos
iVel = c._iVel
iTau = c._iTau

mean = _np.zeros((mstates,dofs))
#mean[iPos, 0] = 0.0
#mean[iVel, 0] = 0.0
cov = _np.zeros((mstates,dofs, mstates,dofs))
dist_initial = (mean, cov)
#cov[iVel,0,iVel,0] = 0.5**2
#cov[iPos,0,iPos,0] = 5.0**2


numIter= int(1.0 / dt)
t = _np.zeros(numIter)
sigmas = _np.zeros((mstates, dofs, numIter))
means = _np.zeros((mstates, dofs, numIter))
dist = dist_initial

dynamicsModel = pandadynamicsmodel.PandaDynamicsModel()

import subprocess
robotDescriptionString = subprocess.check_output(['xacro','/home/roman/catkin_ws/src/franka_ros/franka_description/robots/panda_arm_hand.urdf.xacro'])
urdfModel = pandadynamicsmodel.PandaURDFModel(robotDescriptionString=robotDescriptionString)

timeintegrator = promp.TimeIntegrator(dofs, dynamicsModel=dynamicsModel)

for i in range(numIter):
    dist = timeintegrator.integrate(dist, dt)

    # Jacobian
    urdfModel.setJointPosition(dist[0][iPos])
    jaco_numpy = urdfModel.getEEJacobian()
    
    dist = c.getInstantStateVectorDistribution(currentDistribution=dist, jacobianEE = jaco_numpy)
    t[i] = t[i-1] + dt
    for dof in range(dofs):
        means[:,dof,i] = dist[0][:,dof]
        var = _np.diag(dist[1][:,dof,:,dof])
        sigmas[:,dof,i] = _np.sqrt(var)

dof=5
pylab.figure()    
pylab.plot(t, means[iTau,dof,:])
pylab.plot(t, means[iTau,dof,:]+sigmas[iTau,dof,:])
pylab.plot(t, means[iTau,dof,:]-sigmas[iTau,dof,:])
pylab.title("torque (mean, +-stddev)")

pylab.figure()    
pylab.plot(t, means[iPos,dof,:])
pylab.plot(t, means[iPos,dof,:]+sigmas[iPos,dof,:])
pylab.plot(t, means[iPos,dof,:]-sigmas[iPos,dof,:])
pylab.title("position (mean, +-stddev)")
    
pylab.figure()    
a = 1
pylab.plot(t, means[iVel,dof,:]*a)
pylab.plot(t, (means[iVel,dof,:]+sigmas[iVel,dof,:])*a)
pylab.plot(t, (means[iVel,dof,:]-sigmas[iVel,dof,:])*a)
pylab.title("velocity (mean, +-stddev)")

velFromPos = (means[iPos,dof,1:]-means[iPos,dof,:-1]) / dt
pylab.plot(0.5*t[1:]+0.5*t[:-1], velFromPos, linestyle=":", linewidth=2.0)

c.plot(num=100, distInitial=dist_initial, dynamicsModel=dynamicsModel, urdfModel=urdfModel)

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
