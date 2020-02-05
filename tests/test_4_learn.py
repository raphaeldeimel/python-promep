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


len_derivs = 1
len_dofs = 1

p = promep.ProMeP(index_sizes={
    'dofs': len_dofs, 
    'interpolation_parameters':5, 
    'g': len_derivs,
    'gphi': len_derivs, 
    'gtilde': len_derivs
    }, name='test_4')


from promep import _kumaraswamy

#create a set of "observations:"
num = 30

kv = _np.array((0.0, 0.0))
kp = _np.array((10.0, 1.0))
observations = []
for i in range(30):
    duration = 2.0# + _np.random.normal()
    observed_times = _np.linspace(0, duration, num)
    observed_dts =  ( observed_times[1:] - observed_times[:-1] )
    observed_phases = _np.zeros((num, len_derivs))              # num, g
    observed_Yrefs = _np.zeros((num, 2, len_dofs, len_derivs))   # num,r,d,g
    observed_Xrefs = _np.zeros((num, 2, len_dofs, len_derivs))   # num,rtilde,dtilde,gtilde
    I = _np.eye((2*len_dofs)).reshape((2, len_dofs, 2,len_dofs)) #r,d,rtilde, dtilde
    observed_Ts = _np.tile(I, (num,1,1,1,1))
    
    d_idx = 0

    #set some notrivial phase profile:
    observed_phases[:,0] = _kumaraswamy.cdf(1.45,1.68,_np.linspace(0, 1.0, num))    
    #observed_phases[:,0] = _np.linspace(0, 1.0, num)   
    if len_derivs > 1:
        observed_phases[:,1] = _np.gradient(observed_phases[:,0]) / _np.gradient(observed_times)
    
    observed_phases = observed_phases + 0.01 * _np.random.normal(size=(num, 1))
    velocities = 3.0 * _np.cos(2*observed_times)
    positions = _scipy.integrate.cumtrapz(velocities, x=observed_times, initial=0.0)

    torques   = -0 * positions -15 * velocities
    impulses  = _scipy.integrate.cumtrapz(torques, x=observed_times, initial=0.0)

    #fill into values array:
    data = {
        'position': positions, 
        'velocity': velocities, 
        'torque':   torques, 
        'impulse':  impulses, 
    }
    observed_values = _np.zeros((num, 2, len_dofs, len_derivs)) # num,r,d,g    
    for name in p.readable_names_to_realm_derivative_indices:
        if name in ('kp', 'kv'):
            continue
        r_idx, g_idx = p.readable_names_to_realm_derivative_indices[name]
        observed_values[:,r_idx,d_idx,g_idx] = data[name]
    
    #Xref, Yref remain zero for now
    
    #add observation
    observations.append( (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts) )

    

p.learnFromObservations(observations, max_iterations=10)
p.saveToFile(path='./temp')


p.plot()
p.plot(useTime=False)

p.plotCovarianceTensor()
p.plotExpectedPhase()

if __name__=='__main__':
    import os
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass
    
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
        
        

        
