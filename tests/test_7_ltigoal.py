#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test the basic mixer functionality
@author: Raphael Deimel
"""

import sys
import os
sys.path.insert(0,"../src/")

import numpy as _np
import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

import namedtensors
import mechanicalstate
import staticprimitives



variants = []
for r,g,d in [(2,1,1),(2,2,1),(2,3,1),(2,2,8)]: #test on various combinations of r,g,d parameters

    #make a tensor namespace that hold the right index definitions:
    tns_global = mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=r, g=g, d=d)

    tns_global.registerTensor('CurrentMean', (('r', 'g', 'd'),()) )
    tns_global.registerTensor('CurrentCov', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
    msd_current = mechanicalstate.MechanicalStateDistribution(tns_global, 'CurrentMean', 'CurrentCov')

    ltigoal = staticprimitives.LTIGoal(tns_global, name='test_7_a')
    variants.append( (tns_global, ltigoal, msd_current))

    positions = _np.zeros((d))
    positions[0] = 13.8
    velocities = _np.zeros((d))
    velocities[d-1] = 3.21

    Kv = _np.ones((d,d))
    Kd = _np.arange(d)

    #test various setting methods:
    ltigoal.setDesired()
    ltigoal.setDesired(Kp=10.0)
    ltigoal.setDesired(Kd=Kd)
    ltigoal.setDesired(Kv=Kv)
    ltigoal.setDesired(position=positions)
    ltigoal.setDesired(velocity=velocities)
    ltigoal.setDesired(position=positions, velocity=velocities, torque=42, impulse=_np.cos(_np.linspace(0.1,3.0,d)))

    #test serialization / deserialization:

    serialized = ltigoal.serialize()
    ltigoal_roundtrip_serialize = staticprimitives.LTIGoal.makeFromDict(serialized)

    if ltigoal_roundtrip_serialize.tns.__repr__() != ltigoal.tns.__repr__():
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.name != ltigoal.name:
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.taskspace_name != ltigoal.taskspace_name:
        raise RuntimeError()

        
    filepath = ltigoal.saveToFile(path='./temp')
    ltigoal_roundtrip_saved = staticprimitives.LTIGoal.makeFromFile(filepath)

    if ltigoal_roundtrip_saved.tns.__repr__() != ltigoal.tns.__repr__():
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.name != ltigoal.name:
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.taskspace_name != ltigoal.taskspace_name:
        raise RuntimeError()



##created by ltigoal.serialize()
#reference_serialized = {
# 'Kd': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]],
# 'Kp': [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]],
# 'Kv': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
# 'd': 8,
# 'g': 2,
# 'impulse': [0.9950041652780258,
#  0.8706443108611639,
#  0.5989785432073319,
#  0.22597049350351583,
#  -0.18526991985891028,
#  -0.5651641745205889,
#  -0.849437262400361,
#  -0.9899924966004454],
# 'name': 'test_7_a',
# 'position': [13.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# 'r': 2,
# 'task_space': 'jointspace',
# 'torque': [42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
# 'velocity': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.21]
#}

#if reference_serialized != ltigoal.serialize():
#    raise RuntimeError()

_np.set_printoptions(linewidth=200, precision=4, suppress=True, edgeitems=30)
dofs = ltigoal.tns['d'].size

#Test a larger number of desired goal settings:

K_damping = _np.linspace(-0.5, 3.0, dofs)

timeintegrator = mechanicalstate.TimeIntegrator(tns_global)
dt=0.01
repeats = 5
plotvalues_x = _np.arange(0, 10.0, dt*repeats)
num = plotvalues_x.size
desiredDicts = [{  
    'position':_np.linspace(0.5, 3.5, dofs),
    'velocity': 0.0,
    'impulse': 0.0,
    'torque': 0.0,
    'Kd': 0.0,
    'Kv': K_damping,
    'Kp': 10.0,
    },{
    'position':_np.linspace(0.5, 3.5, dofs),
    'velocity': 0.0,
    'impulse': 0.0,
    'torque': 0.0,
    'Kv': K_damping,
    'Kd': 0.0,
    'Kp': 10.0,
    },{
    'position':0.0,
    'velocity': _np.linspace(-3.5, 3.5, dofs),
    'impulse': 0.0,
    'torque': 0.0,
    'Kv': 2.0,
    'Kd': 0.0,
    'Kp': 0.0,
    },{
    'position':0.0,
    'velocity': _np.linspace(-3.5, 3.5, dofs),
    'impulse': 0.0,
    'torque': 0.0,
    'Kv': 0.1,
    'Kd': 0.0,
    'Kp': 2.0,
    },{
    'position':0.0,
    'velocity': 0.0,
    'impulse': 0.0,
    'torque': 0.0,
    'Kv': 10.0,
    'Kd': 0.0,
    'Kp': 100,
}]

for desiredDict in desiredDicts:
    ltigoal.setDesired(**desiredDict)
    plotvalues_y =_np.zeros((num, dofs))
    msd_current.means.data[...] = 0.0
    for i in range(num):
        msd_expected = ltigoal.getDistribution(msd_current=msd_current)
        timeintegrator.integrate(msd_expected, dt, repeats)
        meansdata  = timeintegrator.msd_current.getMeansData()
        plotvalues_y[i,:] = meansdata[0,0,:]
        msd_current = timeintegrator.msd_current
        

    plt.figure()
    for i in range(ltigoal.tns['d'].size):
        plt.plot(plotvalues_x, plotvalues_y[:,i])



#
# Test different timestep settings of the integrator 
plt.figure()
ltigoal1 = variants[1][1]
msd = variants[1][2]
timeintegrator1 = mechanicalstate.TimeIntegrator(variants[1][0])
ltigoal1.setDesired(position=13.8,  impulse=0.0, torque=0.0, Kp=10, Kv=0.0, Kd=1.0)
for j, (dt, times) in enumerate([(0.33, 1),(0.1, 1),(0.012344, 1),(0.01, 10)]):
    plotvalues_x = _np.arange(0, 3.0, dt*times)
    num = plotvalues_x.size
    plotvalues_y = _np.zeros((num, ltigoal1.tns['d'].size))
    plotvalues_sigma =_np.zeros((num, ltigoal1.tns['d'].size))
    msd.means.data[...] = 0.0
    for i in range(num):
        msd_expected = ltigoal1.getDistribution(msd_current=msd)
        timeintegrator1.integrate(msd_expected, dt, times)
        meansdata  = timeintegrator1.msd_current.getMeansData()
        plotvalues_y[i,:] = meansdata[0,0,:]
        plotvalues_sigma[i,:] = _np.sqrt(timeintegrator1.msd_current.getVariancesData()[0,0,:])
        
        
        msd = timeintegrator1.msd_current
    plt.plot(plotvalues_x, plotvalues_y[:,0], label="{}x{}".format(dt, times))
        
plt.legend()



# test how all mechanical properties behave relative to each other
# also, check evolution of variances

desiredDicts1 = [{  
    'position': 5.0,
    'velocity': 0.0,
    'impulse': 0.0,
    'torque': 0.0,
    'Kd': 3.0,
    'Kv': 0.0,
    'Kp': 10.0,
    },{
    'position':5.0,
    'velocity': 0.0,
    'impulse': 0.0,
    'torque': 0.0,
    'Kv': 0.0,
    'Kd': 0.0,
    'Kp': 30.0,
}]

for desired in desiredDicts1:
    ltigoal1.setDesired(**desired)
    rows = ltigoal1.tns['r'].size * ltigoal1.tns['g'].size
    fig, axesArray = plt.subplots(rows,1, squeeze=False, figsize=(4, 2*rows), sharex='all', sharey='row')
    dt=0.01
    times=1
    plotvalues_x = _np.arange(0, 3.0, dt*times)
    num = plotvalues_x.size
    plotvalues_y = _np.zeros((num, ltigoal1.tns['r'].size,ltigoal1.tns['g'].size,ltigoal1.tns['d'].size))
    plotvalues_sigma =_np.zeros((num, ltigoal1.tns['r'].size,ltigoal1.tns['g'].size,ltigoal1.tns['d'].size))
    msd.means.data[...] = 0.0
    msd.covariances.data_diagonal[...] = 16.0
    for i in range(num):
        msd_expected = ltigoal1.getDistribution(msd_current=msd)
        timeintegrator1.integrate(msd_expected, dt, times)
        meansdata  = timeintegrator1.msd_current.getMeansData()
        msd = timeintegrator1.msd_current
        plotvalues_y[i,:,:,:] = meansdata
        plotvalues_sigma[i,:,:,:] = _np.sqrt(timeintegrator1.msd_current.getVariancesData()) 
        
    d_idx = 0
    for r_idx, r_name in enumerate(ltigoal1.tns['r'].values):    
        for g_idx in ltigoal1.tns['g'].values:    
            row = g_idx + ltigoal1.tns['g'].size * r_idx
            y  = plotvalues_y[:,r_idx, g_idx,d_idx]
            lower = y - 1.95*plotvalues_sigma[:,r_idx,g_idx,d_idx]
            upper = y + 1.95*plotvalues_sigma[:,r_idx,g_idx,d_idx]
            axesArray[row,d_idx].fill_between(plotvalues_x, lower, upper, label="95%",  color=(0.8,0.8,0.8))

            axesArray[row,d_idx].plot(plotvalues_x, lower)
            axesArray[row,d_idx].plot(plotvalues_x, y)
            axesArray[row,d_idx].plot(plotvalues_x, upper)

            axesArray[row,d_idx].set_title(msd.rg_commonnames[r_idx][g_idx])






if __name__=='__main__':
    import common
    common.savePlots()
