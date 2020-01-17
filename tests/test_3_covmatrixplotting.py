#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test whether interface functions work in the default
@author: Raphael Deimel
"""

import sys
import os
sys.path.insert(0,"../src/")
import promep

import numpy as _np
import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

dofs=8
realms=2

p = promep.ProMeP(index_sizes={'dofs': dofs, 'interpolation_parameters':5, 'realms':2, 'derivatives':3}, expected_duration=10., name='test1')

if p.tns.tensorIndices['Wcov'] != (('rtilde','gtilde','stilde','dtilde'),('rtilde_','gtilde_','stilde_','dtilde_')): #make sure we have the same notion of index order
    raise Exception()

Wcov = p.tns.tensorData['Wcov']
Wcov_flat = p.tns.tensorDataAsFlattened['Wcov']

cov_dofs = _np.linspace(-1,1.0,dofs)[:,None] * _np.linspace(-1,1.0,dofs)[None,:]

plot_type='rgsd'
#plot_type='correlations'
#plot_type=None

for i in range(realms):
    for j in range(realms):
        Wcov[i,:,:,:, j,:,:,:] = (i+1)*(j+1)*cov_dofs[None, None, :, None,None,:]


p.plotCovarianceTensor(plot_type=plot_type)

for i in range(3):
    for j in range(3):
        Wcov[:,i,:,:, :,j,:,:] = (i+1)*(j+1)*cov_dofs[None, None, :, None,None,:]

p.plotCovarianceTensor(plot_type=plot_type)

for i in range(3):
    for j in range(3):
        Wcov[:,:,i,:, :,:,j,:] = (i+1)*(j+1)*cov_dofs[None, None, :, None,None,:]

p.plotCovarianceTensor(plot_type=plot_type)

for i in range(dofs):
    for j in range(dofs):
        Wcov[:,:,:,i, :,:,:,j] = (i+1)*(j+1)*dofs**-2

p.plotCovarianceTensor(plot_type=plot_type)


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
