#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2019
@licence: 2-clause BSD licence


"""
import numpy as _np

from . import _namedtensors

class InterpolationGaussian(object):
    """
    This class provides a gaussians-based  interpolation scheme 
    
    """
    def __init__(self, tensornamespace):
        """
        returns a function that takes the phase and outputs the interpolation vector psi

        In other words, it creates a basis function kernel with the given metaparameters
        As proposed the ProMP papers, the basis functions are placed in the interval
        [ -2*sigma ... 1+2*sigma ]

        If you don't specify sigma, it will be guessed so that basis functions overlap reasonably

        serializedDict: If not None, use the dictionary to recreate the serialized interpolation kernel

        
        repeatExtremalSupports: 
            If nonzero, then the first and last basis function are the sums of all basis functions repeated beyond the interval boundaries.
            Effectively, this makes the ProMP converge towards zero velocity close to the boundaries, instead of converging to zero position
            While theoretically, we add an infinite number of basis functions, using the first 3 already is a very good approximation

            This is not done in [1]. For having the original ProMP formulation behavior, set repeatedSupports=0

        """
        self.tensornamespace_parent = tensornamespace
        self.tns = _namedtensors.TensorNameSpace(tensornamespace)
        self.tns.registerTensor('PHI', (('gphi',),('stilde',)))
        
        #some fixed parameters:
        clip_at_sigma = 3.0  #at how many sigmas should the gaussian drop to 0.0? (I.e. to remove its long tail)
        self._repeatExtremalSupports = 3

        #compute with of the gaussians, and where to suppress the tail completely
        delta = 1.0 / (self.tns.indexSizes['stilde']-1)

        self.sigma = 1.0 * delta #no real need to change this to different widths

        self._clipAmount = _np.exp(-(1+clip_at_sigma**2))

        start = 0.0 - self._repeatExtremalSupports * delta
        end   = 1.0 + self._repeatExtremalSupports * delta
        self._phasesOfSupportsAndRepeated = _np.linspace(start, end, self.tns.indexSizes['stilde']+2*self._repeatExtremalSupports)
        self.phasesOfSupports = self._phasesOfSupportsAndRepeated[self._repeatExtremalSupports:-self._repeatExtremalSupports]
        #compute the two fixed factors for the gaussian curves:
        self._a = -0.5/(self.sigma**2)  #in exponent
        self._b = 1.0 
        self._c = -1*self.sigma**-2 #factor for scaling the basis functions for velocities

        #adjust scale once so the inteprolation vector is close to 1.0
        # This is different to [1], where the interpolation vector is always normalized
        # exact unit scale is not necessary though, and without normalizing we maintain a bell curve everywhere
        scale =  _np.sum(self.getPhi(0.5)[0,:])
        self._b = (1.0 + self._clipAmount)  / scale #scale, but also adjust for the clipping offset
        self._clipAmount = self._clipAmount / scale #adjust clipping offset to the scale


    def _evaluateBasisFunctions(self, phase, out_array):
        """
        evaluate the basis function at the given phase
        """
        #compute values of the basis functions:
        distances = _np.clip(phase, 0.0, 1.0) - self._phasesOfSupportsAndRepeated

        Bases0thDerivativeAllUnclipped = self._b * _np.exp(self._a * (distances**2))
        Bases0thDerivativeAll = _np.clip(Bases0thDerivativeAllUnclipped-self._clipAmount, 0.0, _np.inf)
        
        #iterate through all requested derivatives and fill  self.interpolationMatrix:
        nthDerivative = Bases0thDerivativeAll
        for gphi in range(self.tns.indexSizes['gphi']):
            #aggregate the repeated supports beyond the interval into the first and last one:
            out_array[gphi,:] = nthDerivative[self._repeatExtremalSupports:-self._repeatExtremalSupports]
            out_array[gphi,0]  += _np.sum(nthDerivative[:self._repeatExtremalSupports])
            out_array[gphi,-1] += _np.sum(nthDerivative[-self._repeatExtremalSupports:])
            nthDerivative = distances * self._c *  nthDerivative        
        return # data written to out_array 

    def get_PHI_view(self):
        return self.tns.tensorData['PHI'].view()    
            

    def update(self):
        """
        updates the PHI tensor in the calling object's tensor manager
        """
        self._evaluateBasisFunctions(self.tensornamespace_parent.tensorData['phase'][0], out_array = self.tns.tensorData['PHI'])


    def getPhi(self, phase, out_array=None):
        """
        returns interpolation vectors for the mechanical state for each DoF at the given phase

        returns an array of shape:
            gphi x stilde
            
        if out_array is given, write results into this array
        """
        if out_array == None:
            out_array = _np.zeros_like(self.tns.tensorData['PHI'])
        self._evaluateBasisFunctions(phase, out_array)
        return out_array

      