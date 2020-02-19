#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains classes for mixing / coactivating several MSD generators (e.g. ProMeP and PDGoalJointspace)

"""
import warnings as _warnings
import numpy as _np
import itertools as _it
from pprint import pprint as  _pprint

import namedtensors as _namedtensors
import mechanicalstate as _mechanicalstate


class Mixer(object):
    """
    Implements a mixer for mechanical state distributions
    """

    def __init__(self, msd_generator_array, inverseRegularization=1e-9, index_sizes={'r':2, 'g':2, 'd': 8}, max_active_inputs=5):
        """
        
        msd_generator_array: array of generator objects that have a getDistribution() method which returns a MechanicalStateDistribution object
        
        msd_generators are queried for an expected distribution each time the getMixedDistribution() method is called
        
        max_active_inputs: maximum number of generators that can be active at the same time (limits computation time)
        """
        
        self.msd_generators = _np.asarray(msd_generator_array)
        self.active_generators_indices = []
        
        #et up our tensor namespace:
        self.tns = _namedtensors.TensorNameSpace()
        for name in index_sizes:
            self.tns.registerIndex(name, index_sizes[name])
            self.tns.registerIndex(name+'p', index_sizes[name])
        
        self.tns.registerIndex('slots', max_active_inputs)

        self.tensorNameLists = {
          'Mean': ["Mean{}".format(i) for i in range(max_active_inputs)],
          'Cov': ["Cov{}".format(i) for i in range(max_active_inputs)],
        }

        self.tns.registerTensor("preconditioner", (('rp', 'gp', 'dp'),('r', 'g','d')))
        self.tns.registerTranspose("preconditioner", flip_underlines=False)
        self.tns.registerTensor("alpha", (('slots',),()) )
        
        self.tns.registerElementwiseMultiplication('alpha', 'alpha', result_name='alpha_sqaure')
        self.tns.registerSum('alpha', result_name='sumalpha', sumcoordinates=True)
        self.tns.registerSum('alpha_sqaure', result_name='sumalpha2', sumcoordinates=True)

        self.tns.registerTensor("one", ((),()), initial_values='ones')
        self.tns.registerTensor("I", (('rp', 'gp', 'dp'),('r_', 'g_','d_')), initial_values='identity' )

        self.tns.registerTensor("invCovMixed", (('r', 'g', 'd'),('r_', 'g_','d_')), initial_values='identity' )
        self.tns.registerTensor("MeanScaledSum", (('r_', 'g_', 'd_'),()), initial_values='zeros' )
        
        #setup all tensors for each distribution input slot:
        self.tns.registerReset('invCovMixed')
        self.tns.registerReset('MeanScaledSum')
        self._update_order_upto_slot = []
        for slot in range(max_active_inputs):
            self.tns.registerTensor(self.tensorNameLists['Mean'][slot], (('r', 'g', 'd'),()) )
            self.tns.registerTensor(self.tensorNameLists['Cov'][slot], (('r', 'g', 'd'),('r_', 'g_','d_')) )
            
            alphai = self.tns.registerSlice('alpha', {'slots': slot}, result_name='alpha'+str(slot))
            
            previous = self.tns.registerContraction(self.tensorNameLists['Cov'][slot], 'preconditioner') #precondition covariance tensor
            covpreconditioned = self.tns.registerScalarMultiplication(previous, 'sumalpha2')  #adjust for total activation, so to not inflate covariances if sumalpha != 1.0
            
            #compute the regularization factor due to (possibly) not being the only one active:         
            previous = self.tns.registerSubtraction('one', alphai)
            previous = self.tns.registerScalarMultiplication(previous, 'sumalpha')
            lambdai = self.tns.registerScalarMultiplication('I', previous) #Lambda

            previous = self.tns.registerAddition(covpreconditioned, lambdai) #regularize
            previous = self.tns.registerInverse(previous, flip_underlines=False)  #compute precision

            previous = self.tns.registerScalarMultiplication(previous, alphai) #scale precision
            invcovweighted = self.tns.registerContraction( 'preconditioner', previous) #de-precondition
            invcovweightedT = self.tns.registerTranspose(invcovweighted, flip_underlines=False)
            
            #scale means by both precision and activation for computing a sum afterwards
            previous = self.tns.registerContraction(self.tensorNameLists['Mean'][slot], invcovweighted)
            meansscaled = self.tns.registerScalarMultiplication(previous, alphai)

            self.tns.registerAdditionToSlice('invCovMixed',   invcovweightedT, slice_indices={})
            self.tns.registerAdditionToSlice('MeanScaledSum', meansscaled , slice_indices={})

            #in order to avoid unnecessary computation, gather all equations we need to recompute up to the current slot:    
            self._update_order_upto_slot.append(self.tns.update_order.copy() + ['CovMixed', 'MeansMixed'])

        #final computation on the accumulated sums:            
        self.tns.registerInverse('invCovMixed',result_name='CovMixed')
        self.tns.registerContraction('CovMixed', 'MeanScaledSum', result_name='MeanMixed')

        #wrap up everything into an msd object to hand out:
        self.mixed_msd = _mechanicalstate.MechanicalStateDistribution(self.tns, 'MeanMixed','CovMixed', precisionsName='invCovMixed')

        #_pprint(self.tns.tensorIndices)                    

    def mix(self,
            activations,   #nxn matrix 
            phases,        #dxnxn matrix
            current_msd,
            task_spaces = {}
        ):
        """
        compute the mixture of MSD generators by the given activation at the given generalized phase
        
        activations: matrix of activations, expected shape is self.msd_generators.shape
        
        phases: matrix of generalized phases, shape is (self.tns.indexSizes['g']) + self.msd_generators.shape
        
        
        currentDistribution: MechanicalStateDistribution object

        taskspaces: A dictionary of up-to-date taskspace mappings. May be required by some of the generators
                                    
        """
    
        activations = _np.max(0.0, activations) #ensure non-negativitiy to avoid funky behavior
        
        #determine which generators we need to query and mix:
        active_generators_indices = np.argwhere((activations > 0.01 and self.msd_generators != None))
        
        if len(active_generators_indices) > self.max_active_inputs: #for now, just degrade loudly
            warnings.warn("Mixer: more than the maximum number of generators were activated!")
            active_generators_indices = active_generators_indices[:self.max_active_inputs]

        #save a list of active generators, sorted by activation:
        self.active_generators_indices = [indices for activation, indices in sorted(zip(mixedalphas, active_generators_indices))]
        
        #Use the precision tensor of the current msd as preconditioner:
        self.tns.setTensor('preconditioner', current_msd.getPrecisions())
        
        #query the msd_generators:
        last_updated_generator = 0
        for slot, indices in enumerate(self.active_generators_indices):
            msdi = self.msd_generators[indices].getDistribution(generalized_phase=phases[indices], current_msd=current_msd, task_spaces=task_spaces)
            self.tns.setTensor(self.tensorNameLists['Mean'][slot], msdi.getMeansData())
            self.tns.setTensor(self.tensorNameLists['Cov'][slot], msdi.getCovariancesData())

        #mix:
        self.tns.update(self._update_order_upto_slot[len(active_generators_indices)])

        #the result is accessible via the msd_mixed object:    
        return self.msd_mixed


