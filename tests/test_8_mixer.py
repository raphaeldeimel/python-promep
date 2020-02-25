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


#make a tensor namespace that hold the right index definitions:
tns_global = mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=2, g=2, d=1)

tns_global.registerTensor('CurrentMean', (('r', 'g', 'd'),()) )
tns_global.registerTensor('CurrentCov', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
current_msd = mechanicalstate.MechanicalStateDistribution(tns_global, 'CurrentMean', 'CurrentCov')

tns_global.registerTensor('CurrentMean2', (('r', 'g', 'd'),()) )
tns_global.registerTensor('CurrentCov2', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
current_msd2 = mechanicalstate.MechanicalStateDistribution(tns_global, 'CurrentMean2', 'CurrentCov2')


#test1: try to instantiate the default mixer
mixerDefault = mechanicalstate.Mixer(tns_global)



ltigoal0 = staticprimitives.LTIGoal(mixerDefault.tns, Kv=None)
ltigoal1 = staticprimitives.LTIGoal(mixerDefault.tns, Kv=None)

# test 2: create an  as-simple-as-possible mixer

mixer = mechanicalstate.Mixer(tns_global)

#create two lti goal distribution generators:
ltigoal0 = staticprimitives.LTIGoal(tns_global, Kv=None, Kp=100.0)
ltigoal1 = staticprimitives.LTIGoal(tns_global, Kv=None, Kp=10.0)
goal0 = _np.zeros((2,2,1))
goal0[0,0,0] = 13.8
ltigoal0.setDesired(desiredMean=goal0)

goal1 = _np.zeros((2,2,1))
goal1[1,0,0] = -5.555
ltigoal1.setDesired(desiredMean=goal1)


msd_generator_array = _np.array([[ltigoal0, None], [None, ltigoal1]])

activations = _np.zeros((2,2))
activations[0,0] = 0.8
activations[1,1] = 0.2
phases = _np.zeros((2,2))
task_space = None



mixer.mix(msd_generator_array, activations, phases, current_msd, task_space)


timeintegrator = mechanicalstate.TimeIntegrator(tns_global)

dofs=1
dt = 0.01
times =5
plotvalues_x = _np.arange(0, 10.0, dt*times)
num = plotvalues_x.size 
plotvalues_y =_np.zeros((num, 2, 2))
plotvalues_y_sigma = _np.zeros((num, 2, 2))
msd_current = current_msd
plotvalues_activation = _np.zeros((num, 2,2))
for i in range(num):
    
    activations[0,0] = 0.95#_np.clip(-1+3.0 * i / num, 0.0, 1.0)
    activations[1,1] = 1.0 - activations[0,0]
    plotvalues_activation[i,:,:] = activations
    
    mixer.mix(msd_generator_array, activations, phases, msd_current)
    timeintegrator.integrate(mixer.msd_mixed, dt, times)
    msd_current = timeintegrator.msd_current
    
    plotvalues_y[i,:,:] = msd_current.getMeansData()[:,:,0]
    plotvalues_y_sigma[i,:,:] = msd_current.getVariancesData()[:,:,0]
    

fig, axes  = plt.subplots(3,1)
axes[0].plot(plotvalues_x, plotvalues_activation[:,0,0])
axes[0].plot(plotvalues_x, plotvalues_activation[:,1,1])
axes[1].plot(plotvalues_x, plotvalues_y[:,0,0])
axes[1].plot(plotvalues_x, plotvalues_y[:,0,0]+1.95*plotvalues_y_sigma[:,0,0])
axes[1].plot(plotvalues_x, plotvalues_y[:,0,0]-1.95*plotvalues_y_sigma[:,0,0])
axes[2].plot(plotvalues_x, plotvalues_y[:,0,1])
axes[2].plot(plotvalues_x, plotvalues_y[:,0,1]+1.95*plotvalues_y_sigma[:,0,1])
axes[2].plot(plotvalues_x, plotvalues_y[:,0,1]-1.95*plotvalues_y_sigma[:,0,1])
    

if __name__=='__main__':
    import common
    common.savePlots()
