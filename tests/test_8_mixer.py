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
from scipy import signal

import namedtensors
import mechanicalstate
import staticprimitives

from promep import _kumaraswamy

#make a tensor namespace that hold the right index definitions:
tns_global = mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=2, g=2, d=1)

tns_global.registerTensor('CurrentMean', (('r', 'g', 'd'),()) )
tns_global.registerTensor('CurrentCov', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
msd_current = mechanicalstate.MechanicalStateDistribution(tns_global, 'CurrentMean', 'CurrentCov')

tns_global.registerTensor('CurrentMean2', (('r', 'g', 'd'),()) )
tns_global.registerTensor('CurrentCov2', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
msd_current2 = mechanicalstate.MechanicalStateDistribution(tns_global, 'CurrentMean2', 'CurrentCov2')


#test1: try to instantiate the default mixer
mixerDefault = mechanicalstate.Mixer(tns_global)

ltigoalA = staticprimitives.LTIGoal(mixerDefault.tns, Kv=None, name="state_A")
ltigoalB = staticprimitives.LTIGoal(mixerDefault.tns, Kv=None, name="state_B")



# test 2: create an  as-simple-as-possible mixer

mixer = mechanicalstate.Mixer(tns_global)

#create two lti goal distribution generators:
ltigoal1 = staticprimitives.LTIGoal(tns_global, Kp=250.0, Kd=10, name="positiongoal1", expected_torque_noise=0.1)
ltigoal1.setDesired(position=_np.array([[0.5]]))

#ltigoal2 = staticprimitives.LTIGoal(tns_global, Kp=250.0, Kd=-15.5, Kv=0.0, name="positiongoal2", expected_torque_noise=1.1)
ltigoal2 = staticprimitives.LTIGoal(tns_global, Kp=260.0, Kd=15, Kv=0.0, name="positiongoal2", expected_torque_noise=1.01)
ltigoal2.setDesired(position=_np.array([[-0.5]]))

ltigoal3 = staticprimitives.LTIGoal(tns_global, Kp=0.0, Kd=0, Kv=10.0, name="velocitygoal", expected_torque_noise=1.01)
ltigoal3.setDesired(position=_np.array([[-0.0]]), velocity=_np.array([[0.4345]]))

ltigoal4 = staticprimitives.LTIGoal(tns_global, Kp=0.0, Kd=0.0, Kv=0.0, name="float", expected_torque_noise=1.01)


msd_current.covariances.data[...] = 10. * msd_current.covariances.data

activations = _np.zeros((2,2))
activations[0,0] = 0.8
activations[1,1] = 0.2
phases = _np.zeros((2,2))
task_space = None


dofs=1
dt = 0.01
times =1
plotvalues_x = _np.arange(0, 10.0, dt*times)
num = plotvalues_x.size 
print(num)
plotvalues_y =_np.zeros((num, 2, 2))
plotvalues_y_sigma = _np.zeros((num, 2, 2))
msd_current = msd_current
plotvalues_activation = _np.zeros((num, 2,2))
plotvalues_sumalpha = _np.zeros((num, 2))

activation00 = _kumaraswamy.cdf(1.6, 1.8, 0.5+signal.sawtooth(_np.linspace(0, 13. , num), 0.5))
activation11 = _kumaraswamy.cdf(1.6, 1.8, 0.5+signal.sawtooth(_np.linspace(2.95, 18. , num), 0.5))

plotvalues_y_mixed = _np.zeros((num, 2, 2))
plotvalues_generators =_np.zeros((num,2, 2, 2))
plotvalues_generators[...] = _np.nan

for g0, g1, emulate_paraschos in (ltigoal1, ltigoal2, False),(ltigoal2, ltigoal3, False),(ltigoal1, ltigoal4, False),(ltigoal3, ltigoal4, False), (ltigoal1, ltigoal2, True):

    msd_generator_array = _np.array([[g0, None], [None, g1]])
    timeintegrator = mechanicalstate.TimeIntegrator(tns_global, noiseFloorSigmaTorque=0.0, noiseFloorSigmaPosition=0.0, noiseFloorSigmaVelocity=0.0)
    mixer = mechanicalstate.Mixer(tns_global, emulate_paraschos=emulate_paraschos)
    timeintegrator.tns.setTensor('CurrentMean', 0.0)
    timeintegrator.tns.setTensorToIdentity('CurrentCov', scale=0.1**2)
    for i in range(num):
        
        activations[0,0] = activation00[i]
        activations[1,1] = activation11[i]
        if _np.all(activations < 0.011):
            activations[0,0] = 0.011
            
        plotvalues_activation[i,:,:] = activations
        
        
        
        mixer.mix(msd_generator_array, activations, phases, timeintegrator.msd_current)
        timeintegrator.integrate(mixer.msd_mixed, dt, times)


        plotvalues_y[i,:,:] = timeintegrator.msd_current.getMeansData()[:,:,0]
        covariances = timeintegrator.msd_current.getVariancesData()
        plotvalues_y_sigma[i,:,:] = _np.sqrt(covariances[:,:,0])
        
        plotvalues_y_mixed[i,:,:] = mixer.msd_mixed.getMeansData()[:,:,0]

        for k, msd_gen in enumerate(mixer.msds):
            plotvalues_generators[i,k,:,:] = msd_gen.getMeansData()[:,:,0]
        
        asymmetry = _np.sqrt(_np.max((mixer.msd_mixed.covariances.data_flat - mixer.msd_mixed.covariances.data_flat.T)**2))
        if asymmetry > 1e-4:
            print("mixed covariance matrix is asymmetric!: {}".format(asymmetry))

        plotvalues_sumalpha[i,0] = mixer.tns['sumalpha'].data
        plotvalues_sumalpha[i,1] = mixer.tns['sumalpha2'].data


    plotvalues_y[plotvalues_y>1e7] = _np.nan
    plotvalues_y[plotvalues_y<-1e7] = _np.nan

    fig, axes  = plt.subplots(5,1, figsize=(6,6), sharex='all', sharey='row')
    axes[0].set_ylim(0.0, 2.0)
    #axes[1].set_ylim(-1.0, 2.0)
    #axes[2].set_ylim(-3.0, 3.0)
    #axes[3].set_ylim(-100.0, 100.0)
    #axes[4].set_ylim(-20.0, 100.0)    
    if emulate_paraschos:
        fig.suptitle("crossover_paraschos")
    else:
        fig.suptitle("crossover")
    axes[0].set_title('activations')
    axes[0].plot(plotvalues_x, plotvalues_activation[:,0,0], label=g0.name)
    axes[0].plot(plotvalues_x, plotvalues_activation[:,1,1], label=g1.name)
    axes[0].plot(plotvalues_x, plotvalues_sumalpha[:,0], color='b', linestyle=':', label='L1')
    axes[0].plot(plotvalues_x, plotvalues_sumalpha[:,1], color='k', linestyle=(0, (1, 10)), label='L2' )
    axes[0].legend()
    axes[1].set_title('position')
    axes[1].plot(plotvalues_x, plotvalues_y[:,0,0])
    axes[1].plot(plotvalues_x, plotvalues_y_mixed[:,0,0], color='c')
    axes[1].axhline(g0.msd_desired.means.data[0,0,0], linestyle=':')
    axes[1].axhline(g1.msd_desired.means.data[0,0,0], linestyle=':')
    axes[1].fill_between(plotvalues_x, plotvalues_y[:,0,0]-1.95*plotvalues_y_sigma[:,0,0], plotvalues_y[:,0,0]+1.95*plotvalues_y_sigma[:,0,0], label="95%",  color=(0.8,0.8,0.8))


    axes[2].set_title('velocity')
    axes[2].plot(plotvalues_x, plotvalues_y[:,0,1])
    axes[2].plot(plotvalues_x[1:], (plotvalues_y[1:,0,0]-plotvalues_y[:-1,0,0])/dt)
    axes[2].plot(plotvalues_x, plotvalues_y_mixed[:,0,1], color='c')
    axes[2].fill_between(plotvalues_x, plotvalues_y[:,0,1]-1.95*plotvalues_y_sigma[:,0,1], plotvalues_y[:,0,1]+1.95*plotvalues_y_sigma[:,0,1], label="95%",  color=(0.8,0.8,0.8))

    axes[3].set_title('torque')
    axes[3].plot(plotvalues_x, plotvalues_y[:,1,1])
    axes[3].fill_between(plotvalues_x, plotvalues_y[:,1,1]-1.95*plotvalues_y_sigma[:,1,1], plotvalues_y[:,1,1]+1.95*plotvalues_y_sigma[:,1,1], label="95%",  color=(0.8,0.8,0.8))

    axes[4].set_title('impulse')
    axes[4].plot(plotvalues_x, plotvalues_y[:,1,0])
    axes[4].fill_between(plotvalues_x, plotvalues_y[:,1,0]-1.95*plotvalues_y_sigma[:,1,0], plotvalues_y[:,1,0]+1.95*plotvalues_y_sigma[:,1,0], label="95%",  color=(0.8,0.8,0.8))
        


if __name__=='__main__':
    import common
    common.savePlots()
