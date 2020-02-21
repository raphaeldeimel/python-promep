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

import namedtensors
import mechanicalstate
import staticprimitives


for r,g,d in [(2,1,1),(2,2,1),(2,3,1),(2,2,8)]: #test on various combinations of r,g,d parameters

    #make a tensor namespace that hold the right index definitions:
    tns_global = mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=r, g=g, d=d)

    tns_global.registerTensor('CurrentMean', (('r', 'g', 'd'),()) )
    tns_global.registerTensor('CurrentCov', (('r', 'g', 'd'),('r_', 'g_', 'd_')), initial_values='identity' )
    current_msd = mechanicalstate.MechanicalStateDistribution(tns_global, 'CurrentMean', 'CurrentCov')

    ltigoal = staticprimitives.LTIGoal(tns_global, name='test_7_a')

    positions = _np.zeros((d))
    positions[0] = 13.8
    velocities = _np.zeros((d))
    velocities[d-1] = 3.21
    print(velocities)

    Kv = _np.ones((d,d))
    Kd = _np.arange(d)

    #test various setting methods:
    ltigoal.setDesired()
    ltigoal.setDesired(Kp=10.0)
    ltigoal.setDesired(Kd=Kd)
    ltigoal.setDesired(Kv=Kv)
    ltigoal.setDesired(position=positions)
    ltigoal.setDesired(velocity=velocities)
    ltigoal.setDesired(position=positions, velocity=velocities, torque=42, impulse=_np.cos(_np.linspace(0.1,3.0,d)))

    #test serialization / deserialization:

    serialized = ltigoal.serialize()
    ltigoal_roundtrip_serialize = staticprimitives.LTIGoal.makeFromDict(serialized)

    if ltigoal_roundtrip_serialize.tns.__repr__() != ltigoal.tns.__repr__():
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.name != ltigoal.name:
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.taskspace_name != ltigoal.taskspace_name:
        raise RuntimeError()

        
    filepath = ltigoal.saveToFile(path='./temp')
    ltigoal_roundtrip_saved = staticprimitives.LTIGoal.makeFromFile(filepath)

    if ltigoal_roundtrip_saved.tns.__repr__() != ltigoal.tns.__repr__():
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.name != ltigoal.name:
        raise RuntimeError()
    if ltigoal_roundtrip_serialize.taskspace_name != ltigoal.taskspace_name:
        raise RuntimeError()

    

#if __name__=='__main__':
#    import common
#    common.savePlots()
