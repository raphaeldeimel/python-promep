#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test the trajectory compositors
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


import promep._trajectorycompositors 

from promep import _kumaraswamy



len_derivs = 2
len_dofs = 1

for repeatExtremal in (0, 3):
    for len_trajectoryparameters in (3, 5, 11):

        p = promep.ProMeP(index_sizes={
            'dofs': len_dofs, 
            'interpolation_parameters':len_trajectoryparameters, 
            'g': len_derivs,
            'gphi': len_derivs, 
            'gtilde': len_derivs
            }, name='test_4')
        p.trajectorycompositor.changeExtremalRepetition(repeatExtremal)

        num=100

        
        observed_phases = _kumaraswamy.cdf(1.45,1.68,_np.linspace(0, 1.0, num))    
        observed_phases = _np.linspace(0.0, 1.0, num)    

        tc = _np.array([p.trajectorycompositor.getPhi(phase) for phase in observed_phases])

        for g in (0,1):
            plt.figure()
            plt.suptitle("stilde{}repeatextremal{}".format(len_trajectoryparameters,repeatExtremal ))
            for s in range(len_trajectoryparameters):
                plt.plot(observed_phases, tc[:,g,s])

            plt.plot(observed_phases, _np.sum(tc[:,g,:], axis=1), linestyle=':')
            
            plt.plot(p.trajectorycompositor.phasesOfSupports, [-0.02]*len_trajectoryparameters, marker='o', linewidth=0 )

if __name__=='__main__':
    import common
    common.savePlots()
        
        

        
