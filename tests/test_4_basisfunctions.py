#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test the interpolation kernels for proper functionality & plot them

The figures should show a set of gaussian-like functions shifted along the phase in equal increments.
The sums (dotted lines) should perfectly sum up to 1 for positions and 0 for velocities

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
_np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.3f}'.format})
import matplotlib.pylab as pylab



counts=[4,5,20]

md = promp.MechanicalStateDistributionDescription(dofs=1)  


x = _np.linspace(0,1,1000)
for count in counts:
    ik = promp.InterpolationKernelGaussian(md, count)
    ys = _np.stack([ik.getInterpolationVectorPosition(c) for c in x])
    ysD = _np.stack([ik.getInterpolationVectorVelocity(c) for c in x])

    pylab.figure()
    pylab.title("Basis functions for position")
    pylab.plot(x,ys)
    pylab.plot(x,_np.sum(ys, axis=1), linestyle=":", label="sum")
    pylab.legend()
    pylab.figure()
    pylab.title("Basis functions for velocity")
    pylab.plot(x,ysD)
    pylab.plot(x,_np.sum(ysD, axis=1), linestyle=":", label="sum")
    pylab.legend()
    #make sure the interpolation vectors have the right properties for all phase values:
    s = _np.sum(ys, axis=1)
    if  _np.any( _np.abs(s-1.0) > 1e-2):
        print("Failed: interpolation vector for positions should sum to 1.0 for count={0}".format(count))
    s = _np.sum(ysD, axis=1)    
    if _np.any( _np.abs(s) > 1e-2):
        print("Failed: interpolation vector for velocities should sum to 0.0 count={0}".format(count))

    #try out the interpolation vectors on an example support vector:



count = 3
y_end=10.0
for y_start, y_end in ((0, 10.0),(10.0, 10.0)):
    w = _np.linspace(y_start,y_end,count)
    ik_original = promp.InterpolationKernelGaussian(md, count, extendBy=0.0, repeatExtremalSupports=0)
    ik = promp.InterpolationKernelGaussian(md, count, extendBy=0.0, repeatExtremalSupports=3)
    position = _np.stack(   [_np.dot(ik.getInterpolationVectorPosition(c), w) for c in x] )
    velocity = _np.stack(   [_np.dot(ik.getInterpolationVectorVelocity(c), w) for c in x] )
    position_original = _np.stack(   [_np.dot(ik_original.getInterpolationVectorPosition(c), w) for c in x] )
    velocity_original = _np.stack(   [_np.dot(ik_original.getInterpolationVectorVelocity(c), w) for c in x] )
    pylab.figure()
    #pylab.title("results for trajectory with linearly increasing weights/supports")
    pylab.plot(x,position_original, label=r"Position $\varphi$")
    pylab.plot(x,position, label=r"Position $\varphi'$")
    pylab.plot(ik.phasesOfSupports,w, label=None, marker='X', linewidth=0)

    pylab.plot(x,velocity_original, label=r"Velocity $\varphi$")
    pylab.plot(x,velocity, label=r"Velocity $\varphi'$")
#    pylab.plot(0.5*(x[1:]+x[:-1]),(position[1:]-position[:-1])/(x[1:]-x[:-1]), label="velocity according to position", linestyle=':')
    legend = pylab.legend()
    pylab.xlabel(r"Phase")
    pylab.ylabel(r"$\Psi \cdot ({0},{1},{2})$".format(w[0], w[1],w[2]))



if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
