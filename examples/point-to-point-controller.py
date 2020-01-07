#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:19:06 2017

This test creates a ProMP to pass a point with specified velocity


@author: raphael
"""


import sys
sys.path.insert(0,"../src/")
import promp

import matplotlib.pylab as pylab

mp1 = promp.ProMPFactory.makePositionVelocityController( 
                    goalPositions=[1.0], 
                    goalVelocities=[3.0], 
                    
                 currentPositions=[-2.0], 
                currentVelocities=[-0.5], 
            currentPositionSigmas=[0.5], 
            currentVelocitySigmas=[0.0]
)


mp1.plot(withSampledTrajectories=15)

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)

