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
trajectory_parameters = 7

p = promep.ProMeP(index_sizes={
    'dofs': len_dofs, 
    'interpolation_parameters':trajectory_parameters, 
    'g': len_derivs,
    'gphi': len_derivs, 
    'gtilde': len_derivs,
    }, name='test_6a')

modelfreepromp = promep.ProMeP(index_sizes={
    'dofs': len_dofs, 
    'interpolation_parameters':trajectory_parameters, 
    'g': len_derivs,
    'gphi': len_derivs, 
    'gtilde': len_derivs,
    }, name='test_6b')


from promep import _kumaraswamy

#create a set of "observations:"
num = 100
duration = 1.0

kv = _np.array((0.0, 0.0))
kp = _np.array((10.0, 1.0))
observations = []
free_variables = []
for i in range(30):
    duration = 2.0 + _np.random.random() * 16.0
    duration = 10.0
    offset = 5.0 * _np.random.normal() 
    free_variables.append((duration, offset))
    observed_times = _np.linspace(0, duration, num)
    observed_dts =  ( observed_times[1:] - observed_times[:-1] )
    observed_phases = _np.zeros((num, len_derivs))              # num, g
    observed_values = _np.zeros((num, 2, len_derivs, len_dofs)) # num,r,g,d
    observed_Yrefs = _np.zeros((num, 2, len_derivs, len_dofs))   # num,r,g,d
    observed_Xrefs = _np.zeros((num, 2, len_derivs, len_dofs))   # num,rtilde,gtilde,dtilde
    I = _np.eye((2*len_dofs)).reshape((2, len_dofs, 2,len_dofs)) #r,d,rtilde, dtilde
    observed_Ts = _np.tile(I, (num,1,1,1,1))
    
    d_idx = 0

    #set some notrivial phase profile:
    observed_phases[:,0] = _kumaraswamy.cdf(2.0,2.0,_np.linspace(0, 1.0, num))    
    observed_phases[:,1] = _np.gradient(observed_phases[:,0]) / _np.gradient(observed_times)
    
    
    goal = _np.random.normal()*10
    positions = -5. + 10.0 * observed_phases[:,0]
    velocities = _np.gradient(positions) / _np.gradient(observed_times)

#    torques_kp = -30 * positions
#    torques_kv = -30 * velocities
#    torques   = torques_kp  + _np.random.random(num) #+5*_np.cos(_np.linspace(0, num, num) + 6.28 *_np.random.random())
#    #torques   = 10.0 * observed_phases[:,0] + 2 * (_np.random.random(num)-0.5)
    impulses  = 3 * (positions -  positions[0])
    torques = _np.gradient(impulses) / _np.gradient(observed_times)

    #fill into values array:
    data = {
        'position': positions, 
        'velocity': velocities, 
        'torque':   torques, 
        'impulse':  impulses, 
    }
    for name in p.commonnames2rg:
        if len(p.commonnames2rg[name])!=2:
            continue
        r_idx, g_idx = p.commonnames2rg[name]
        observed_values[:,r_idx,g_idx,d_idx] = data[name]
    
    #Xref, Yref remain zero for now
    
    #add observation
    observations.append( (observed_times, observed_phases,observed_values, observed_Xrefs,observed_Yrefs, observed_Ts) )

    
mask = [
    {'rtilde': 'effort', 'gtilde': 0}, #no scale-free impulse trajectory
    {'rtilde': 'effort', 'gtilde': 1}, #no scale-free torque trajectory
    {'rtilde': 'motion', 'gtilde': 1}, #no scale-free velocity trajectory
]  # =model-free promp equivalent

p.learnFromObservations(observations)
modelfreepromp.learnFromObservations(observations, mask=mask)


plotRanges = {
    'position': 30, 
    'velocity': 3.0, 
    'impulse': 100, 
    'torque': 20, 
}
plotRangesPhase = {
    'position': 30, 
    'velocity': 30, 
    'impulse': 100, 
    'torque': 200, 
}

whatToPlot=['position', 'velocity', 'torque', 'impulse']

p.plot(whatToPlot=whatToPlot, plotRanges=plotRanges)
p.plot(useTime=False, whatToPlot=whatToPlot, plotRanges=plotRangesPhase)
p.plotCovarianceTensor()



modelfreepromp.plot(whatToPlot=whatToPlot, plotRanges=plotRanges)
modelfreepromp.plot(useTime=False, whatToPlot=whatToPlot, plotRanges=plotRangesPhase)
modelfreepromp.plotCovarianceTensor(omit_masked_parameters=True)


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
        
        

        
