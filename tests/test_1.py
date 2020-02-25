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



p_different = promep.ProMeP(index_sizes={'gphi':1}) #create promep where all indices used are of different size


#promps only model motion, 
promp = promep.ProMeP(index_sizes={'r': 2, 'rtilde':2, 'g': 2, 'gphi':2, 'gtilde':1 })
promp = promep.ProMeP(index_sizes={'derivatives': 1, 'realms': 1})

promp_modelfree = promep.ProMeP(index_sizes={ 'g': 2, 'gphi':1, 'gtilde':3 })

dofs = 4
derivatives = 3
p = promep.ProMeP(index_sizes={'dofs': dofs, 'interpolation_parameters':3, 'realms':2, 'derivatives':derivatives}, expected_duration=10., name='test1')

Wmean = _np.zeros(p.tns['Wmean'].shape)
rtilde = 0
gtilde = 0
for dtilde in range(p.tns['dtilde'].size):
    Wmean[rtilde,gtilde,:,dtilde] += 0.3456 * _np.cos(_np.linspace(0, 3.0*dtilde, p.tns['stilde'].size))

rtilde = 1
gtilde = 2
for dtilde in range(p.tns['dtilde'].size):
    Wmean[rtilde,gtilde,:,dtilde] += - 5 * dtilde
    Wmean[rtilde,1,:,dtilde] += _np.linspace(0, 15, p.tns['stilde'].size) 

Wcov_flat = _np.zeros(p.tns['Wcov'].shape_flat)
for i in range(Wcov_flat.shape[0]):
    Wcov_flat[i,i] = 5.0
Wcov = Wcov_flat.reshape(p.tns['Wcov'].shape)
#r,g,s,d
for s in range(3):
    Wcov[1,2,s,:, 0,0,s,:] = -1.0 * (s+2) * _np.eye((dofs))
    Wcov[0,0,s,:, 1,2,s,:] = Wcov[1,2,s,:, 0,0,s,:]


p.setParameterDistribution(Wmean, Wcov)

sampled = p.sample()
generalized_phase = [0.89, 1.0, 0.0][:derivatives]
dist = p.getDistribution(generalized_phase=generalized_phase)
dist.plotCorrelations()

p.plot(addExampleTrajectories=None)
p.plotCovarianceTensor()



m = p.getDistribution(generalized_phase=[0.5, 1.0, 1.0])



#test serialization, writing, reading and deserialization:
try:
    os.mkdir('temp')
except FileExistsError:
    pass

p.saveToFile(path='temp/')
p2 = promep.ProMeP.makeFromFile('temp/test1.promep.h5')
if p2.name != p.name:
    raise Exception()
if p2.expected_duration != p.expected_duration:
    raise Exception()
if p2.tns.index_names != p.tns.index_names:
    raise Exception()
if _np.any(p2.tns['Wmean'].data != p.tns['Wmean'].data): #bit-perfect?
    raise Exception()
if _np.any(p2.tns['Wcov'].data != p.tns['Wcov'].data):  #bit-perfect?
    raise Exception()

if (p2.tns.__repr__() != p.tns.__repr__()):  #same repr?
    raise Exception()
    


if __name__=='__main__':
    import common
    common.savePlots()
