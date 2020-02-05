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

from promep import _kumaraswamy

#create a set of "observations:"
num = 30

max_error = 1e-2

for a_target, b_target in [ (1.0, 1.0), (2.0, 2.0), (3.0, 1.5), (0.5, 0.7), (1.455, 1.688) ]:
    observations = _np.empty((num, 2))
    observations[:,0] = _np.linspace(0.0, 1., num) 
    observations[:,1] = _kumaraswamy.cdf(a_target,b_target,observations[:,0])
    a,b,rms,i = _kumaraswamy.approximate(observations, accuracy=1e-5, max_iterations=100000)
    if ((a_target - a)**2 + (b_target - b)**2)**0.5 > max_error:
        raise Exception("Approximation failed for parameters {},{}".format(a_target, b_target))

#plot the last test:
plt.plot(observations[:,0], observations[:,1])
plt.plot(observations[:,0], _kumaraswamy.cdf(a,b,observations[:,0]), linestyle=':')


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
        
        

        
