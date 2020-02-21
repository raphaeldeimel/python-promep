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
    
    ti = mechanicalstate.TimeIntegrator(tns_global)
    

#if __name__=='__main__':
#    import common
#    common.savePlots()
