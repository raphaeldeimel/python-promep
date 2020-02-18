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


p = promep.ProMeP.makeFromFile('temp/test_2.promep.h5')

p.plot()
p.plot(useTime=False)

p.plotCovarianceTensor()





if __name__=='__main__':
    import common
    common.savePlots()
        
