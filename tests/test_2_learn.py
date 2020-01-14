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
import matplotlib.pylab as pylab

import matplotlib.pyplot as plt


len_derivs = 1
len_dofs = 1
p = promep.ProMeP(index_sizes={'dofs': len_dofs, 'interpolation_parameters':3, 'g': len_derivs, 'gphi': len_derivs, 'gtilde': 1}, expected_duration=10.0, name='test_2')


from promep import _kumaraswamy

#create a set of "observations:"
num = 10
duration = 1.0

kv = _np.array((0.0, 0.0))
kp = _np.array((10.0, 1.0))
observations = []
free_variables = []
for i in range(10):
    duration = (_np.random.random() + 1.5) * 5.0
    offset = _np.random.random() - 0.5
    free_variables.append((duration, offset))
    
    observed_phases = _np.zeros((num, len_derivs))              # num, g
    observed_values = _np.zeros((num, 2, len_dofs, len_derivs)) # num,r,d,g
    observed_Yrefs = _np.zeros((num, 2, len_dofs, len_derivs))   # num,r,d,g
    observed_Xrefs = _np.zeros((num, 2, len_dofs, len_derivs))   # num,rtilde,dtilde,gtilde
    I = _np.eye((2*len_dofs)).reshape((2, len_dofs, 2,len_dofs)) #r,d,rtilde, dtilde
    observed_Ts = _np.tile(I, (num,1,1,1,1))
    
    #set some notrivial phase profile:
    observed_phases[:,0] = _kumaraswamy.cdf(1.5,1.5,_np.linspace(0,1,num))
    if len_derivs > 1:
        observed_phases[1:,1] = (observed_phases[1:,0] - observed_phases[:-1,0]) * (num/duration)
    
    #positions = 0th order motion
    observed_values[:,0,0,0] = 10.0 * observed_phases[:,0] + offset + _np.random.random(num) / num

    observed_values[:,1,0,0] = 100 * _np.random.random(num) 


    if len_derivs > 1:
        # 0th order effort = torques. simulate spring+damper system:
        observed_values[:,1,:,0] = kp[None,:len_dofs] * observed_values[:,0,:,0] + kv[None,:len_dofs] * observed_values[:,0,:,0]  

        #compute 1st derivatives empirically:
        observed_values[1:,:,:,1] = ( observed_values[1:,:,:,0] - observed_values[:-1,:,:,0] ) * (num/duration)
        observed_values[0,:,:,1] = observed_values[1,:,:,1]


    
    #Xref, Yref remain zero for now
    
    #add observation
    observations.append( (observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts) )


figures = [plt.figure() for i in range(4)]
for fig in figures:
    fig.add_axes([0,0,1,1])

for (observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts),(duration, offset) in zip(observations, free_variables):
    time = _np.linspace(0, duration, num)
    figures[0].axes[0].plot(time, observed_phases[:,0])
    #figures[0].axes[0].plot(time, observed_phases[:,1]) 
    #figures[1].axes[0].plot(observed_phases[:,0], observed_phases[:,1]) 
    figures[2].axes[0].plot(time, observed_values[:,0,0,0]) #r,d,g
    #figures[2].axes[0].plot(time, observed_values[:,0,0,1]) #r,d,g
    figures[3].axes[0].plot(observed_phases[:,0], observed_values[:,0,0,0]) #r,d,g
    figures[3].axes[0].plot(observed_phases[:,0], observed_values[:,1,0,0]) #r,d,g
    #figures[3].axes[0].plot(observed_phases[:,0], observed_values[:,0,0,1] / observed_phases[:,1] ) #r,d,g


p.learnFromObservations(observations)

pylab.figure()
pylab.plot(p.negLL[10:])
pylab.figure()
relative_changes = (p.negLL[1:]-p.negLL[:-1])/p.negLL[:-1]
pylab.plot(relative_changes)
pylab.ylim(-0.01, 0.01)

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
        
        

        
