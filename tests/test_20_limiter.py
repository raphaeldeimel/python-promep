#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../src/")
import promp

import promp._tensorfunctions as _t

import numpy as _np
import matplotlib
import matplotlib.animation
import matplotlib.pylab as pylab

md = promp.MechanicalStateDistributionDescription(dofs=2,derivativesCountEffort=0)

limiter = promp.limiter.Limiter(md)



means = _np.random.random((2,3))
stds  =_np.random.random((2,3))**2+0.2
covs  = stds[:,:,None, None] * stds[None, None,:,:]



md = promp.MechanicalStateDistributionDescription(dofs=3, derivativesCountEffort=0)
limiter = promp.limiter.Limiter(md)

limits_pos = _np.zeros((2,3))
limits_pos[0,:] = -0.123
limits_pos[1,:] = 1.234
limits_vel = _np.zeros((2,3))
limits_vel[0,:] = -0.555
limits_vel[1,:] = 0.666
limiter.setLimits(limits_pos, limits_vel)

means = _np.array([[1.,-1.,2.],[0.1,0.7,-1.3]])

stddev = _np.array([[0.5,1.,1.],[0.1, 0.01,100.]])
covs  = _t.I(2,3,lam=stddev)


means_limited, covs_limited = limiter.limit((means, covs))

print(_np.sqrt(_t.getDiagView(covs_limited)))

if  _np.sum((covs_limited - _t.T(covs_limited)**2)) > 1e-8:
    print("Test ERROR: returned covariance tensor is not symmetric!")
    



###
md = promp.MechanicalStateDistributionDescription(dofs=1, derivativesCountEffort=0 )
limiter = promp.limiter.Limiter(md)

limits_pos = _np.zeros((2,1))
limits_pos[0,:] = -1.234
limits_pos[1,:] = 1.234
limits_vel = _np.zeros((2,1))
limits_vel[0,:] = -0.555
limits_vel[1,:] = 0.666
limiter.setLimits(limits_pos, limits_vel)

means = _np.array([[0.],[0.1]])

stddev = _np.array([[2.5],[0.1]])
covs  = _t.I(2,1,lam=stddev**2)



def createfig():
    n=100
    xlimits = (-2.,2.)
    xdata = _np.linspace(xlimits[0], xlimits[1], n)
    fig,ax = pylab.subplots()
    ln, = pylab.plot([], [], 'r', animated=True)
    vline = ax.axvline([0.0])
    ax.axvline(limits_pos[0,0])
    ax.axvline(limits_pos[1,0])
    ax.set_xlim(xlimits[0],xlimits[1])
    ax.set_ylim(0, 2.0)
    return fig, ax, xdata, ln, vline
    
pylab.show()

def plotframe(x):
    means[0,0] = x
    means_limited, covs_limited = limiter.limit((means, covs))
    var = covs_limited[0,0,0,0]
    mu = means_limited[0,0]
    print(mu-x)
    ydata = 1 * _np.e**(-0.5*(xdata-mu)**2/var) * (2*_np.pi*var)**-0.5  #gaussian
    ln.set_data(xdata, ydata)
    vline.set_data( [[mu], [mu]], [0,2.0] )

    return ln,vline,

fig, ax, xdata, ln, vline = createfig()
ani = matplotlib.animation.FuncAnimation(fig, plotframe, frames=_np.linspace(-1.3, 1.3, 100), blit=True, interval=100, repeat_delay=1000)
ani.save('./plots/limits.mp4', writer=matplotlib.animation.writers['ffmpeg'](fps=15, metadata=dict(artist='python-promp'), bitrate=1800))


for x in [-1.3, 0.0, 0.7, 1.3, 1.5]:
    fig, ax, xdata, ln, vline = createfig()
    plotframe(x)



#n=100
#xvalues = _np.linspace(-1.30, 1.30, n)
#yvalues = _np.zeros((n))
#for i,x in enumerate(xvalues):
#    means[0,0] = x
#    means_limited, covs_limited = limiter.limit((means, covs))
#    yvalues[i] = _np.sqrt(covs_limited[0,0,0,0])
#pylab.plot(xvalues, yvalues)
#pylab.axhline(stddev[0,0], linestyle='dotted')

if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
