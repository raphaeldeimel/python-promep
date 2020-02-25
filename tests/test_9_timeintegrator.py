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

import mechanicalstate


#for r,g,d in [(2,1,1),(2,2,1),(2,3,1),(2,2,8)]: #test on various combinations of r,g,d parameters
for r,g,d in [(2,2,8)]: #test on various combinations of r,g,d parameters

    #make a tensor namespace that hold the right index definitions:
    tns_global = mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=r, g=g, d=d)

    tns_global.registerTensor('LastMean', (('r', 'g', 'd'),()) )
    tns_global.registerTensor('LastCov', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
    last_msd = mechanicalstate.MechanicalStateDistribution(tns_global, 'LastMean', 'LastCov')
    tns_global['LastMean'].data[0,0,:] = 1.0
    tns_global['LastMean'].data[0,1,:] = 1.0
    
    ti = mechanicalstate.TimeIntegrator(tns_global)
    dt = 0.01
    
    current_msd = ti.integrate(last_msd, dt)    

#if __name__=='__main__':
#    import common
#    common.savePlots()
