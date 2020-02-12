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

        #set up tensor namespaces for each active generator:
        self.tns_per_generator = []
        for i in range(max_active_inputs):
            tns_local = _namedtensors.TensorNameSpace(self.tns) #inherit indices
            #input tensors/scalars:
            tns_local.registerTensor("preconditioner", (('r', 'g', 'd'),('r_', 'g_','d_')))
            tns_local.registerTensor("Cov", (('r', 'g', 'd'),('r_', 'g_','d_')))
            tns_local.registerTensor("Mean", (('r', 'g', 'd'),()) )
            tns_local.registerTensor("alphai", ((),()) )
            tns_local.registerTensor("sumalpha", ((),()) )
            tns_local.registerTensor("sumalpha2", ((),()) )
            #some constants we need:
            tns_local.registerTensor("one", ((),()))
            tns_local.setTensor("one", 1.0)
            tns_local.registerTensor("I", (('r', 'g', 'd'),('r_', 'g_','d_')), initial_values='identity' )
            
            #compute the regularization factor due to (possibly) not being the only one active:         
            tns_local.registerSubtraction('one', 'alphai')
            tns_local.registerScalarMultiplication('(one-alphai)', 'sumalpha')
            tns_local.registerScalarMultiplication('I', '((one-alphai)*sumalpha)', result_name='Lambda')
            
            #precondition, regularize, invert, weigh, and de-precondition:
            tns_local.registerContraction('Cov', 'preconditioner', result_name='C') #precondition covariance tensor
            tns_local.registerScalarMultiplication('C', 'sumalpha2')  #adjust for total activation, so to not inflate covariances if sumalpha != 1.0
            tns_local.registerAddition('(C*sumalpha2)', 'Lambda', 'Covregularized') #regularize
            tns_local.registerInverse('Covregularized', 'invCov')  #compute precision
            tns_local.registerScalarMultiplication('invCov', 'alpha') #scale precision
            tns_local.registerContraction( 'preconditioner', '(invCov*alpha)', result_name='invCovWeighted') #de-precondition
            
            #scale means by both precision and activation for computing a sum afterwards
            tns_local.registerContraction('Mean', 'invCov')
            tns_local.registerScalarMultiplication('invCov:alpha', 'alphai', 'MeanScaled')
            
            
            tns_local.tns_per_generator.append(tns_local)

        
        self.tns.registerTensor("invCovMixed", (('r', 'g', 'd'),('r_', 'g_','d_'))) #sum of all invCovWeighted
        self.registerInverse('invCovMixed',result_name='CovMixed')
        self.tns.registerTensor("MeansScaledSum", (('r', 'g', 'd'),()))   #sum of all "MeanScaled"
        self.registerContraction('invCovMixed', 'MeansScaledSum', result_name='MeansMixed')



    def getMixedDistribution(self,
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

        if len(active_generators_indices) > self.max_active_inputs: #for now, just fail loudly
            warnings.warn("Mixer: more than the maximum number of generators were activated!")
            active_generators_indices = active_generators_indices[:self.max_active_inputs]

    
        #compute global scaling values:
        mixedalphas = [activations[indices] for indices in active_generators_indices]
        sumalpha = _np.sum(mixedalphas)
        sumalpha2 = _np.sum(_np.array(mixedalphas)**2)
        
        #save a list of active generators, sorted by activation:
        self.active_generators_indices = [indices for activation, indices in sorted(zip(mixedalphas, active_generators_indices))]
        
        
        #Use the current precision tensor as preconditioner:
        preconditioner = current_msd.getPrecisions()
        
        #query the msd_generators:
        last_updated_generator = 0
        for i, indices in enumerate(self.active_generators_indices):
                msdi = self.msd_generators[indices].getDistribution(generalized_phase=phases[indices], current_msd=current_msd, task_spaces=task_spaces)
                #transfer data into local tensor namespace:
                tns_generator = self.tns_per_generator[i]
                tns_generator.setTensor("Cov", msdi.means)
                tns_generator.setTensor("Mean", msdi.covariances )
                tns_generator.setTensor('alphai', activations[indices])
                tns_generator.setTensor('sumalpha', sumalpha)
                tns_generator.setTensor('sumalpha2', sumalpha2)
                tns_generator.setTensor('preconditioner', preconditioner)
                tns_generator.update()
                last_active_generator = i
        
        #aggregate results:
        invCovWeighted = []
        meansScaled = []
        for tns_generator in self.tns_per_generator[:last_updated_generator+1]:
            invCovWeighted.append(tns_generator.tensorData['invCovWeighted'])
            meansScaled.append(tns_generator.tensorData['MeanWeighted'])
        
        #finish computatopm of the mixed distribution:
        self.tns.setTensor("invCovMixed", _np.sum(invCovWeighted))
        self.tns.setTensor("MeansScaledSum", _np.mean(MeansScaled))
        self.tns.update() #updates MeansMixed and CovMixed
        
        #copy result into a new msd object and hand it back:
        msd_mixed =  MechanicalStateDistribution(self.tns.tensorData['MeansMixed'],self.tns.tensorData['CovMixed'], self.tns.tensorData['invCovMixed'])
        return msd_mixed


