#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test the learning function
@author: Raphael Deimel
"""

import sys
import os
sys.path.insert(0,"../src/")
import promep

import numpy as _np
import scipy as _scipy

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt


len_derivs = 2
len_dofs = 1

p = promep.ProMeP(index_sizes={
    'dofs': len_dofs, 
    'interpolation_parameters':5, 
    'g': len_derivs,
    'gphi': len_derivs, 
    'gtilde': len_derivs,
    }, name='test_2')


from promep import _kumaraswamy

#create a set of "observations:"
num = 100
duration = 1.0

kv = _np.array((0.0, 0.0))
kp = _np.array((10.0, 1.0))
observations = []
free_variables = []
for i in range(30):
    duration = (_np.random.random() + 0.1) * 10.0
    offset = 1.0 * (_np.random.random() - 0.5)
    free_variables.append((duration, offset))
    
    observed_times = _np.linspace(0, duration, num)
    observed_dts =  ( observed_times[1:] - observed_times[:-1] )
    observed_phases = _np.zeros((num, len_derivs))              # num, g
    observed_values = _np.zeros((num, 2, len_derivs,  len_dofs)) # num,r,d,g
    observed_Yrefs = _np.zeros((num, 2, len_derivs, len_dofs))   # num,r,d,g
    observed_Xrefs = _np.zeros((num, 2,len_derivs, len_dofs))   # num,rtilde,dtilde,g
    I = _np.eye((2*len_dofs)).reshape((2, len_dofs, 2,len_dofs)) #r,d,rtilde, dtilde
    observed_Ts = _np.tile(I, (num,1,1,1,1))
    
    d_idx = 0

    #set some notrivial phase profile:
    observed_phases[:,0] = _kumaraswamy.cdf(2.0,2.0,_np.linspace(0, 1.0, num))    
    if len_derivs > 1:
        observed_phases[:,1] = _np.gradient(observed_phases[:,0]) / _np.gradient(observed_times)
    
    positionerror = offset + 0.1*_np.random.random(num) / _np.sqrt(num)
    positions = 10.0 * observed_phases[:,0] + positionerror
    velocities = _np.gradient(positions) / _np.gradient(observed_times)

    torques_kp = -30 * positions
    torques   = (torques_kp + _np.random.random(num)) / duration
    impulses  = -30 * observed_phases[:,0]
    torques  =_np.gradient(impulses) / _np.gradient(observed_times)
    
    _scipy.integrate.cumtrapz(torques, x=observed_times, initial=0.0) 

    #fill into values array:
    data = {
        'position': positions, 
        'velocity': velocities, 
        'torque':   torques, 
        'impulse':  impulses, 
    }
    for name in p.commonnames2rg:
        if name in ('kp', 'kv'):
            continue
        r_idx, g_idx = p.commonnames2rg[name]
        observed_values[:,r_idx,g_idx,d_idx] = data[name]
    
    #Xref, Yref remain zero for now
    
    #add observation
    observations.append( (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts) )

    

p.learnFromObservations(observations, max_iterations=500)
p.saveToFile(path='./temp')

#p.plotLearningProgress()

p.plot()
p.plot(useTime=False)

p.plotCovarianceTensor()





if __name__=='__main__':
    import common
    common.savePlots()
