#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2019
@licence: 2-clause BSD licence

This file contains the common definitions for a distribution over mechanical state

"""
import numpy as _np
from . import _tensorfunctions as _tens

class MechanicalStateDistributionDescription(object):
    def __init__(self, dofs=1, derivativesCountMotion=2, derivativesCountEffort=1, serializedDict=None):
        self.mStateNamesPossible = ['torque', 'position','velocity','acceleration','jerk']
        self.mStateDerivativesPossible = [1,           0,         1,             2,     3]
        
        if serializedDict is not None:
            self.deserialize(serializedDict)
        else:
            self.derivativesCountMotion = derivativesCountMotion
            self.derivativesCountEffort = derivativesCountEffort
            self.mStateNames = self.mStateNamesPossible[1-self.derivativesCountEffort:1+self.derivativesCountMotion]
            self.mStateDerivatives = self.mStateDerivativesPossible[1-self.derivativesCountEffort:1+self.derivativesCountMotion]
            self.dofs = dofs        
            self.mechanicalStatesCount = self.derivativesCountEffort + self.derivativesCountMotion
            self._update()

    def _update(self):
        #check:
        if self.derivativesCountMotion > 2:
            raise NotImplementedError()
        if self.derivativesCountEffort > 1:
            raise NotImplementedError()

        #create a list of names for each mstate, so they can be uniquely identified
        self.mStateNames2Index = {} #reverse lookup index->names
        for i, name in enumerate(self.mStateNames):
            self.mStateNames2Index[name] = i
        
        #create a reverse lookup derivative->mstateIndicesDerivatives 
        self.mStateDerivatives2Indices = {}  
        for i, deriv in enumerate(self.mStateDerivatives):
            if not deriv in self.mStateDerivatives2Indices:
                self.mStateDerivatives2Indices[deriv] = []
            self.mStateDerivatives2Indices[deriv].append(i)

        #create a dictionary of lists of indices for each time derivative:
        self.mStateIndicesDerivatives = {}
        for d in range(0,4):
            self.mStateIndicesDerivatives[d] = [ self.mStateNames2Index[name] for (name, deriv) in zip(self.mStateNames, self.mStateDerivatives) if deriv==d ]
        #create a list of indices that change sign if time flow reverts:
        self.mStateIndicesTimeDirectionDependent = [ self.mStateNames2Index[name] for (name, deriv) in zip(self.mStateNames, self.mStateDerivatives) if deriv%2==1 ] #gather all mstates that change direction if time direction changes
        

    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this class

        """
        serializedDict = {}
        serializedDict[u'motion derivatives'] = self.derivativesCountMotion
        serializedDict[u'effort derivatives'] = self.derivativesCountEffort
        serializedDict[u'mechanical state names'] = self.mStateNames
        serializedDict[u'mechanical state derivation orders'] = self.mStateDerivatives
        serializedDict[u'n dofs'] = self.dofs
        return serializedDict


    def deserialize(self, serializedDict):
        """
        sets the internal parameters according to the given dictionary

        serializedDict: dictionary created with self.serialize()
        """
        self.derivativesCountMotion = serializedDict[u'motion derivatives']
        self.derivativesCountEffort = serializedDict[u'effort derivatives']
        self.mStateNames = serializedDict[u'mechanical state names']
        self.mStateDerivatives = serializedDict[u'mechanical state derivation orders']
        self.dofs = serializedDict["n dofs"]
        self.mechanicalStatesCount = self.derivativesCountEffort + self.derivativesCountMotion
        self._update()
        
    def __eq__(self, other):
        if self.derivativesCountMotion != other.derivativesCountMotion:  return False
        if self.derivativesCountEffort != other.derivativesCountEffort:  return False
        if self.mStateNames            != other.mStateNames:             return False
        if self.mStateDerivatives      != other.mStateDerivatives:       return False
        if self.dofs                   != other.dofs:                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
