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

realms=2
derivatives=3
interpolation_parameters=5
dofs=7
p = promep.ProMeP(index_sizes={'dofs': dofs, 'interpolation_parameters':interpolation_parameters, 'realms':realms, 'derivatives':derivatives}, expected_duration=10., name='test1')

if p.tns.tensorIndices['Wcov'] != (('rtilde','gtilde','stilde','dtilde'),('rtilde_','gtilde_','stilde_','dtilde_')): #make sure we have the same notion of index order
    raise Exception()

Wcov = p.tns.tensorData['Wcov']
Wcov_flat = p.tns.tensorDataAsFlattened['Wcov']

cov_dofs = _np.linspace(-1,1.0,dofs)[:,None] * _np.linspace(-1,1.0,dofs)[None,:]

for normalize in ('', 'rg', 'rgsd'):

    for i in range(realms):
        for j in range(realms):
            Wcov[i,:,:,:, j,:,:,:] = (i+1)*(j+1)*cov_dofs[None, None, :, None,None,:]

    p.plotCovarianceTensor(normalize)

    for i in range(derivatives):
        for j in range(derivatives):
            Wcov[:,i,:,:, :,j,:,:] = (i+1)*(j+1)*cov_dofs[None, None, :, None,None,:]

    p.plotCovarianceTensor(normalize)

    for i in range(interpolation_parameters):
        for j in range(interpolation_parameters):
            Wcov[:,:,i,:, :,:,j,:] = (i+1)*(j+1)*cov_dofs[None, None, :, None,None,:]

    p.plotCovarianceTensor(normalize)

    for i in range(dofs):
        for j in range(dofs):
            Wcov[:,:,:,i, :,:,:,j] = (i+1)*(j+1)*dofs**-2

    p.plotCovarianceTensor(normalize)




if __name__=='__main__':
    import common
    common.savePlots()
