#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test of task space stiffness

@author: roman
"""

import sys
sys.path.insert(0,"../src/")
import promp
import promp._tensorfunctions as _t

import pandadynamicsmodel


import numpy as _np
import matplotlib.pylab as pylab
import matplotlib 
print(matplotlib.matplotlib_fname())



#try to change this:
dofs = 8

# controller parameters, for a specific timestep
desiredPosition = [-1.8401137043108495, 0.30112427659482943, 2.2994922044716244,
    -1.532449399303914, -0.16456990414214756, 1.3626471059095324, 0.6593281112074011,
    0.0799377356350053]
kp = [300.0, 300.0, 300.0, 100.0, 100.0, 100.0] 
kv = [50.0, 50.0, 50.0, 10.0, 10.0, 10.0]
massMatrixInverse = _np.eye(1) / 0.4567
dt = 0.010


import subprocess
robotDescriptionString = subprocess.check_output(['xacro','/home/roman/ws_phastapromp/src/franka_ros/franka_description/robots/panda_arm_hand.urdf.xacro'])
urdfModel = pandadynamicsmodel.PandaURDFModel(robotDescriptionString=robotDescriptionString)

c = promp.TaskSpaceController(kp=kp,kv=kv,dofs=dofs,desiredPosition=desiredPosition,urdfModel=urdfModel)
mstates = c._md.mechanicalStatesCount
mStateNames = c._md.mStateNames
iPos = c._iPos
iVel = c._iVel
iTau = c._iTau

mean = _np.zeros((mstates,dofs))
mean[iPos, :] = [-0.3, 0.30, 1.3,
    -1.0, -0.16, 0.36, 0.659,
    0.08]
#mean[iVel, :] = 10.0
#mean[iVel, 0] = 0.0

initialVariances = _np.dot(_np.array([0.01,0.1,0.1])[:,_np.newaxis], _np.ones((1,dofs)))
dist_initial = [   
    mean,
    _t.makeCovarianceTensorUncorrelated(mstates, dofs, initialVariances)
] 

numIter= int(10.0 / dt)
t = _np.zeros(numIter)
sigmas = _np.zeros((mstates, dofs, numIter))
means = _np.zeros((mstates, dofs, numIter))
errs = _np.zeros((mstates, 6, numIter))
dist = dist_initial

dynamicsModel = pandadynamicsmodel.PandaDynamicsModel()

timeintegrator = promp.TimeIntegrator(dofs, dynamicsModel = dynamicsModel)

for i in range(numIter):
    dist = timeintegrator.integrate(dist, dt)
    maxmean = dist[0].max()
    argmaxmean =  _np.unravel_index(_np.argmax(dist[0]), (3,8))
    maxconv = dist[1].max()
    argmaxconv =  _np.unravel_index(_np.argmax(dist[1]), (3,8,3,8))

    # Jacobian
    urdfModel.setJointPosition(dist[0][iPos])
    jacoUrdf = urdfModel.getJacobian()
    hTransformUrdf = urdfModel.getEELocation()
    
    dist = c.getInstantStateVectorDistribution(currentDistribution=dist)
    for dofTS in range(6):
        errs[:,dofTS,i] = c.errTS[:,dofTS]

    t[i] = t[i-1] + dt
    for dof in range(dofs):
        means[:,dof,i] = dist[0][:,dof]
        var = _np.diag(dist[1][:,dof,:,dof])
        #with _np.errstate(invalid='raise'):
        sigmas[:,dof,i] = _np.sqrt(var)

dof=0
pylab.figure()
pylab.plot(t, means[iTau,dof,:])
pylab.plot(t, means[iTau,dof,:]+sigmas[iTau,dof,:])
pylab.plot(t, means[iTau,dof,:]-sigmas[iTau,dof,:])
pylab.title("torque (mean, +-stddev)")
#pylab.show(block=True)

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

pylab.figure()    
pylab.plot(t, errs[iPos,0,:], label='x')
pylab.plot(t, errs[iPos,1,:], label='y')
pylab.plot(t, errs[iPos,2,:], label='z')
pylab.title("task space error")
pylab.legend()

#velFromPos = (means[iPos,dof,1:]-means[iPos,dof,:-1]) / dt
#pylab.plot(0.5*t[1:]+0.5*t[:-1], velFromPos, linestyle=":", linewidthc = promp.TaskSpaceController(kp=kp,kv=kv,dofs=doc = promp.TaskSpaceController(kp=kp,kv=kv,dofs=dofs,desiredPosition=desiredPosition,urdfModel=urdfModel)fs,desiredPosition=desiredPosition,urdfModel=urdfModel)=2.0)
c2 = promp.TaskSpaceController(kp=kp,kv=kv,dofs=dofs,desiredPosition=desiredPosition,urdfModel=urdfModel)
c2.plot(num=100, duration = 1.0, linewidth=0.3,withSampledTrajectories=1,distInitial=dist_initial, urdfModel=urdfModel, dynamicsModel=dynamicsModel)

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        fig = pylab.figure(n)
        pylab.figure(n).savefig(filename,  bbox_extra_artists=(pylab.figure(n).get_axes()[0],))
