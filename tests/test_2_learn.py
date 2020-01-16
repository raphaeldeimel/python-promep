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
p = promep.ProMeP(index_sizes={'dofs': len_dofs, 'interpolation_parameters':7, 'g': len_derivs, 'gphi': len_derivs, 'gtilde': len_derivs}, name='test_2')



from promep import _kumaraswamy

#create a set of "observations:"
num = 100
duration = 1.0

kv = _np.array((0.0, 0.0))
kp = _np.array((10.0, 1.0))
observations = []
free_variables = []
for i in range(30):
    duration = (_np.random.random() + 1.5) * 5.0
    offset = 1.0 * (_np.random.random() - 0.5)
    free_variables.append((duration, offset))
    
    observed_times = _np.linspace(0, duration, num)
    observed_dts =  ( observed_times[1:] - observed_times[:-1] )
    observed_phases = _np.zeros((num, len_derivs))              # num, g
    observed_values = _np.zeros((num, 2, len_dofs, len_derivs)) # num,r,d,g
    observed_Yrefs = _np.zeros((num, 2, len_dofs, len_derivs))   # num,r,d,g
    observed_Xrefs = _np.zeros((num, 2, len_dofs, len_derivs))   # num,rtilde,dtilde,gtilde
    I = _np.eye((2*len_dofs)).reshape((2, len_dofs, 2,len_dofs)) #r,d,rtilde, dtilde
    observed_Ts = _np.tile(I, (num,1,1,1,1))
    
    d_idx = 0

    #set some notrivial phase profile:
    observed_phases[:,0] = _kumaraswamy.cdf(2.0,2.0,_np.linspace(0, 1.0, num))    
    if len_derivs > 1:
        observed_phases[:,1] = _np.gradient(observed_phases[:,0]) / _np.gradient(observed_times)
       
    positions = 10.0 * observed_phases[:,0] + offset + _np.random.random(num) / _np.sqrt(num)
    velocities = _np.gradient(positions) / _np.gradient(observed_times)
    torques   = 10.0 * observed_phases[:,0] + 2 * (_np.random.random(num)-0.5)
    impulses  = _scipy.integrate.cumtrapz(torques, x=observed_times, initial=0.0)

    #fill into values array:
    data = {
        'position': positions, 
        'velocity': velocities, 
        'torque':   torques, 
        'impulse':  impulses, 
    }
    for name in p.readable_names_to_realm_derivative_indices:
        if name in ('kp', 'kv'):
            continue
        r_idx, g_idx = p.readable_names_to_realm_derivative_indices[name]
        observed_values[:,r_idx,d_idx,g_idx] = data[name]
    
    #Xref, Yref remain zero for now
    
    #add observation
    observations.append( (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts) )


pylab.figure()
for (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts),(duration, offset) in zip(observations, free_variables):
    pylab.plot(observed_times, observed_phases[:,0])

pylab.figure()
for (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts),(duration, offset) in zip(observations, free_variables):
    pylab.plot(observed_times, observed_values[:,0,0,0])
    pylab.plot(observed_times, observed_values[:,1,0,0])


if len_derivs >=2:
    pylab.figure()
    for (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts),(duration, offset) in zip(observations, free_variables):
        pylab.plot(observed_phases[:,0], observed_phases[:,1])

    

p.learnFromObservations(observations, max_iterations=500)
p.saveToFile(path='./temp')

pylab.figure()
pylab.plot(p.negLL)
pylab.xlabel("Iteration")
pylab.ylabel("-log( p(training data|model) )")
pylab.ylim(0, 10 * _np.mean(p.negLL[-10:]))

p.plot()
p.plot(useTime=False)

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
        
        

        
