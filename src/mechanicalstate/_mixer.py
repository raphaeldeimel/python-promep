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

    def __init__(self, tns, max_active_inputs=5, force_product_of_distributions=False, force_no_preconditioner=False):
        """
        
        tns: a TensorNameSpace object containing the index definitions to use:
        
        'r': realms index
        'g': time derivatives index
        'd': degrees of freedom index
        
        max_active_inputs: maximum number of generators that can be active at the same time (limits computation time)
        

        The following two arguments should not be set, unless you want to demonstrate degraded performance:
        
        force_product_of_distributions: if true, compute the mixture like it is usually done in ProMP papers
        force_no_preconditioner: Force the preconditioner matrix to be the identity
        
        """
        
        self.active_generators_indices = []
        
        #et up our tensor namespace:
        self.tns = _namedtensors.TensorNameSpace(tns)
        self.tns.cloneIndex('r', 'rp')
        self.tns.cloneIndex('g', 'gp')
        self.tns.cloneIndex('d', 'dp')
        
        self.tns.registerIndex('slots', max_active_inputs)

        self.tensorNameLists = {
          'Mean': ["Mean{}".format(i) for i in range(max_active_inputs)],
          'Cov': ["Cov{}".format(i) for i in range(max_active_inputs)],
        }

        self.tns.registerTensor("invCovMinimum", (('r_', 'g_', 'd_'),('r', 'g','d')))
        self.tns.setTensorToIdentity('invCovMinimum', 0.001)

        self.tns.registerTensor("MeanScaledSum", (('r_', 'g_', 'd_'),()), initial_values='zeros' )
        self.tns.registerTensor("invCovMixed", (('r_', 'g_', 'd_'),('r', 'g','d')), initial_values='identity' ) #inverse of mixed covariance of last iteration


        self.tns.registerScalarMultiplication('invCovMixed', 1.0, result_name = 'preconditioner_wrongindices') #use the last computed invCovMixed as preconditioner
        self.tns.renameIndices('preconditioner_wrongindices', {'r_': 'rp', 'g_': 'gp', 'd_':'dp'}, result_name = 'preconditioner') #use the last computed invCovMixed as preconditioner
        
        if force_product_of_distributions:
            self.tns.registerReset('preconditioner', 'identity') #precondition covariance tensor


        self.tns.registerTensor('alpha', (('slots',),()), initial_values='zeros' )
        
        self.tns.registerElementwiseMultiplication('alpha', 'alpha', result_name='alpha_sqaure')
        self.tns.registerSum('alpha', result_name='sumalpha', sumcoordinates=True)
        self.tns.registerSum('alpha_sqaure', result_name='sumalpha2', sumcoordinates=True)

        self.tns.registerTensor("one", ((),()), initial_values='ones')
        self.tns.registerTensor("I", (('rp', 'gp', 'dp'),('r_', 'g_','d_')), initial_values='identity' )

        
        #setup all tensors for each distribution input slot:
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
            if force_product_of_distributions:  #skip adding the lambdai (decorrelating) term
                previous = covpreconditioned            
            previous = self.tns.registerInverse(previous, flip_underlines=False)  #compute precision

            previous = self.tns.registerScalarMultiplication(previous, alphai) #scale precision
            invcovweighted = self.tns.registerContraction( 'preconditioner', previous, result_name='invCovWeighted{}'.format(slot)) #de-precondition
            invcovweightedT = self.tns.registerTranspose(invcovweighted, flip_underlines=False)
            
            #scale means by both precision and activation for computing a sum afterwards
            meansscaled = self.tns.registerContraction(self.tensorNameLists['Mean'][slot], invcovweighted, result_name = 'meansscaled{}'.format(slot))


            #in order to avoid unnecessary computation, gather all equations we need to recompute up to the current slot:    
            self._update_order_upto_slot.append(list(self.tns.update_order))
            #self._update_order_upto_slot.append(self.tns.update_order.copy() + ['CovMixed_unsym', 'CovMixed_unsym2', '(CovMixed_unsym2)^T','CovMixed', 'MeanMixed'])

            include_everything_after = len(self.tns.update_order)


        #aggregate all mixed         
        self.tns.registerReset('invCovMixed', 'zeros')
        self.tns.registerReset('MeanScaledSum', 'zeros')
        #self.tns.registerAdditionToSlice('invCovSum',   'invCovMinimum' , slice_indices={})

        for slot in range(max_active_inputs):
            a1 = self.tns.registerAdditionToSlice('invCovMixed',   'invCovWeighted{}'.format(slot) , slice_indices={})
            a2 = self.tns.registerAdditionToSlice('MeanScaledSum', 'meansscaled{}'.format(slot) , slice_indices={})
            self._update_order_upto_slot[slot] = self._update_order_upto_slot[slot] + self.tns.update_order[include_everything_after:]

        include_everything_after = len(self.tns.update_order)


        #final computation on the accumulated sums:            
        previous = self.tns.registerInverse('invCovMixed', result_name='CovMixed', flip_underlines=False)

        self.tns.registerContraction('CovMixed', 'MeanScaledSum', result_name='MeanMixed')

        #wrap up everything into an msd object to hand out:
        self.msd_mixed = _mechanicalstate.MechanicalStateDistribution(self.tns, 'MeanMixed','CovMixed', precisionsName='invCovMixed')

        for slot in range(max_active_inputs):
            self._update_order_upto_slot[slot] = self._update_order_upto_slot[slot] + self.tns.update_order[include_everything_after:]


    def mix(self,
            msd_generator_array,
            activations,   #nxn matrix 
            phases,        #dxnxn matrix
            current_msd,
            task_spaces = {}
        ):
        """
        compute the mixture of MSD generators by the given activation at the given generalized phase
        
        msd_generator_array: array of generator objects to query. (method getDistribution() will be called)
        
        activations: matrix of activations, expected shape is self.msd_generators.shape
        
        phases: matrix of generalized phases, shape is (self.tns['g'].size) + self.msd_generators.shape
        
        
        currentDistribution: MechanicalStateDistribution object

        taskspaces: A dictionary of up-to-date taskspace mappings. May be required by some of the generators
                                    
        """
        if msd_generator_array.shape != activations.shape:
            raise ValueError("First three arguments must be of same shape! msd_generator_array is {}, while activations is {}".format(msd_generator_array.shape,activations.shape))
        if msd_generator_array.shape != phases.shape:
            raise ValueError("First three arguments must be of same shape! msd_generator_array is {}, while phases is {}".format(msd_generator_array.shape,phases.shape))
    
        activations = _np.maximum(0.0, activations) #ensure non-negativitiy to avoid funky behavior
        
        #determine which generators we need to query and mix:
        generator_exists = (msd_generator_array != None)
        generator_is_active = (activations > 0.01)
        active_generators_indices_unsorted = _np.argwhere(_np.logical_and(generator_is_active,generator_exists))

        alpha = [activations[tuple(indices)] for indices  in active_generators_indices_unsorted]
        #save a list of active generators, sorted by activation:
        ranking = sorted(zip(alpha, active_generators_indices_unsorted), reverse=True, key=lambda pair: pair[0])
        active_generators_indices = [tuple(indices) for activation, indices in ranking]
        
        if len(active_generators_indices) > self.tns['slots'].size: #for now, just degrade loudly
            warnings.warn("Mixer: more generators than available mixing slots were activated!")
            active_generators_indices = active_generators_indices[:self.tns['slots'].size] #ignore the lesser activated generators
            alpha = alpha[:self.tns['slots'].size]

        if len(active_generators_indices) == 0: #nothing is active! raise error to avoid safety issues
            raise ValueError("Mixer warning: no generator is active!")

        self.active_generators_indices = active_generators_indices
        self.active_generators_indices_unsorted = active_generators_indices_unsorted

        self.tns.resetTensor('alpha', 'zeros') #make sure that all unused slot activations are zero - just in case
        #query the msd_generators:
        last_updated_generator = 0
        self.msds = []
        for slot, indices in enumerate(self.active_generators_indices):
            generatori = msd_generator_array[tuple(indices)]
            activationi = activations[tuple(indices)]
            phasei = phases[tuple(indices)]
            msdi = generatori.getDistribution(generalized_phase=phasei, msd_current=current_msd, task_spaces=task_spaces)
            self.tns.setTensor(self.tensorNameLists['Mean'][slot], msdi.getMeansData())
            self.tns.setTensor(self.tensorNameLists['Cov'][slot], msdi.getCovariancesData())
            self.tns['alpha'].data[slot] = activationi-0.01
            self.msds.append(msdi)

        #mix:
        self.tns.update(self._update_order_upto_slot[len(self.active_generators_indices)-1])



        #the result is accessible via the msd_mixed object:    
        return self.msd_mixed


