#!/bin/ipython

import numpy
import numpy as _np

from minimalvariance import *



#quick test of limit cases should be close to 0.2 except the last one:
print(minimal_variances_3(0.2, 0.2, 0.2))
print(minimal_variances_3(0.2, 0.2,  10))
print(minimal_variances_3(0.2,  10,  10))
print(minimal_variances_3( 10,  10, 0.2))
print(minimal_variances_3( 10, 0.2,  10))
print(minimal_variances_3( 10,  10,  10))

#same for recursive implementation:
print(minimal_variances([0.2, 0.2, 0.2], lambda x: x**-1))
print(minimal_variances([0.2, 0.2,  10], lambda x: x**-1))
print(minimal_variances([0.2,  10,  10], lambda x: x**-1))
print(minimal_variances([ 10, 0.2,  10], lambda x: x**-1))
print(minimal_variances([ 10,  10,  10], lambda x: x**-1))


#visualize the decision landscape for a simple case: three 1D distributions:

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


n=201
x=numpy.linspace(0.01, 10.0, n)[:,None]
y = numpy.linspace(0.01, 10.0, n)[None,:]
xgrid = x*numpy.ones((1,n))
xgrid_b = numpy.ones((n,1)) * y
levels = numpy.array([0.1, 1, 2, 5,9]) #isoline levels
#levels = numpy.linspace(0.01,10.0,5) #isoline levels

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(xgrid, xgrid_b, minimal_variances_deimel(x, y), levels, colors='b')  #iterative method
#ax.contour3D(xgrid, xgrid_b,0.5*(x+y),levels, colors='k') #just for comparison: simple linear combination
ax.contour3D(xgrid, xgrid_b, minimal_variances_3(x, y,factors=(1,-2,3)),levels, colors='r') #better approximation of taking the minimum  variance
ax.contour3D(xgrid, xgrid_b, minimal_variances_3(x, y,factors=(1,-1.,3)),levels, colors='g') #converges quicker but underestimates variances when inputs are the same

#ax.contour3D(xgrid, xgrid.T, minimal_variances_3( x, y,y), levels, colors='r') #influence of a third input
#ax.contour3D(xgrid, xgrid.T, minimal_variances_3(x, y,y, factors=(0.5,0,0)),levels, colors='m')   #promp-style mixing, assuming zero correlation
ax.contour3D(xgrid, xgrid_b, func_ref_min(x, y),levels, colors='y') #that's what we are aiming for


#code for plotting ellipses:

import numpy as _np
import matplotlib.pyplot as plt
import matplotlib.patches

def plot_cov_ellipse(cov, pos=[0.0, 0.0], nstds=[0.5,1.0,2.0], axes=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstds : The radii of the ellipses in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = _np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if axes is None:
        axes = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = _np.degrees(_np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    ellipses = []
    for sigma in nstds:
        width, height = 2 * sigma * _np.sqrt(vals)
        lw=3.0/sigma**1
        if width < 0.5*lw and height < 0.5*lw:
            e = matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, joinstyle='round', fill=True, lw=0.0, **kwargs)            
        else:
            e = matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, joinstyle='round', fill=False,lw=lw, **kwargs)
        ellipses.append(e)
        axes.add_artist(e)
    return ellipses


S1 = _np.array([[0.720,0.70],[0.70, 0.720]])
S2 = _np.array([[1,0],[0,1]])
S3 = _np.array([[0.720,-0.700],[-0.700, 0.720]])
S4 = _np.array([[0.860,-0.600],[-0.600, 0.450]])

def plot_comparison_ellipses(S1, S2, lim=2.5):
    fig=plt.figure()
    plot_cov_ellipse(S1, [0.0,0.0], color='#AA0000', alpha=0.3)
    plot_cov_ellipse(S2, [0.0,0.0], color='#000088', alpha=0.3)
#    plot_cov_ellipse(minimal_variances([S1, S2]), [0.0,0.0], color='#000000', alpha=0.3)
    plot_cov_ellipse(mixing_deimel(S1, S2), color='#000000', alpha=0.3)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

#plot_comparison_ellipses(S1, S2)
#plot_comparison_ellipses(S1, S3)
#plot_comparison_ellipses(S2, S3)
#plot_comparison_ellipses(S3, S4)

plt.ion()
plt.show()






