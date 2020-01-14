#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2020
@licence: 2-clause BSD licence

This file contains the main code for computing tensors that map from a task space to the common configuration space (joint space)
"""

import numpy as _np
from . import _namedtensors


class JointSpaceToJointSpaceTransform(object):
    """
    Simplest possible mapping: Joint-space to Joint-space
    
    If update() is called, it reads in the tensor Yref and sets tensors Xref and T
    """
    
    def __init__(self, tensornamespace):
        self.tns = _namedtensors.TensorNameSpace(tensornamespace) #copies index definitions only
        #Path to compute T:
        self.tns.registerTensor('Yref', (('r','d','g',),()), external_array=tensornamespace.tensorData['Yref'] ) #where to linearize
        self.tns.registerTensor('Jt', (('d',),('dtilde',)))
        self.tns.registerTensor('Jinv', (('d',),('dtilde',)))
        self.tns.registerBasisTensor('e_motion_motion', (('r',),('rtilde',)), (('motion',), ('motion',)) )
        self.tns.registerContraction('e_motion_motion', 'Jinv')
        if not 'effort' in self.tns.indexValues['r']: #this is in case we emulate ProMPs which don't model the effort domain at all
            self.tns.registerTensor('e_effort_effort:Jt', (('r', 'd'),('rtilde', 'dtilde'))) #placeholde tensor with only zeros in it
        else: 
            self.tns.registerBasisTensor('e_effort_effort', (('r',),('rtilde',)), (('effort',), ('effort',)) )
            self.tns.registerContraction('e_effort_effort', 'Jt')

        #computed output tensors:
        self.tns.registerTensor('Xref', (('rtilde','dtilde','g'),()), external_array=tensornamespace.tensorData['Xref']  )      
        self.tns.registerAddition('e_effort_effort:Jt', 'e_motion_motion:Jinv', result_name='T', out_array=tensornamespace.tensorData['T']) #has indices (('r', 'd')('rtilde', 'dtilde'))

        self.tns.setTensorToIdentity('Jt')
        self.tns.setTensorToIdentity('Jinv')

        #conceptually; to also consider dotJ:
        #self.tns.registerTensor('dotJinv', (('d',),('dtilde',)))
        #self.tns.registerBasisTenso('e_motion_1_motion_0', (('r',),('rtilde',)), (('motion',), ('motion',)) )
        #self.tns.registerContraction('e_motion_1_motion_0', 'dotJinv')        
        #self.tns.registerAddition('e_effort_effort:Jt+e_motion_motion:Jinv', 'e_motion_1_motion_0:dotJinv', result_name='T') #has indices (('r', 'd')('rtilde', 'dtilde'))
        self.update()
        
    def update(self):
        # for other mappings: set Xref, Jt and Jinv here
        self.tns.setTensor('Xref', self.tns.tensorData['Yref']) #for joint to joint space, simply copy the reference point
        self.tns.update() #recomputes T


