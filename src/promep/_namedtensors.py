#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2019
@licence: 2-clause BSD licence

This file contains the main code for representing ProMePs, visualizing them, and
also to sample trajectories from a ProMeP

"""

import numpy as _np


class NamedTensorsManager(object):
    """
    This is kind of a band-aid class to augment current numpy/tensorflow/pytorch versions with 
    named indices as well as providing the notion of upper and lower indices for proper tensor contraction semantics
    
    
    Should hopefully become superfluous at some point in the future.
    
    Note: pytorch > 1.3.1 has experimental support for named indices, but they don't have a notion of upper/lower indices 


    General concept:
    
        Numpy's tensordot function requires us to specify matching index pairs as tuples of dimension numbers of the arrays representing the input tensors
        
        This helper function takes in two tensor names, a dictionary of tensor index tuples, and computes the axes parameter of numpy.tensordot, 
        as well as the index names of the result tensorDotAxesTuples

        Tensor indices are specified as a tuple of two tuples, representing the order of upper and lower indices of the tensor respectively.

        The implicit assumption is, that numpy's array dimensions relate to a concatenation of upper indices (first) and lower indices (last).
        
        Example #1: tensor $T^a^b_c_d$:
            Numpy array has 4 dimensions
            tuples that map tensor indices: ('a','b'), ('c','d')
        Example #2: tensor $A^e^f:
            Numpy array has 2 dimensions
            tuples that map tensor indices: ('e','f'), ()


    Implementation Choices:
        * For tensor notation, the order of indices is irrelevant. We require matching index orders when doing binary operations though to simplify implementation
    
    """

    def __init__(self):
        self.tensorIndices = {}
        self.tensorIndexPositions = {}
        self.indexSizes = { }
        self.indexValues = { }
        self.tensorData = {}            # contains the actual array 
        self.tensorDataAsFlattened = {} # flattened view on the actual array
        self.tensorShape = {}
        self.tensorShapeFlattened = {}
        self.registeredAdditions = {}
        self.registeredSubtractions = {}
        self.registeredContractions = {}
        self.registeredTransposes = {}
        self.registeredInverses = {}
        self._tensordot = _np.tensordot #which library to use
        self.update_order = []


    def registerIndex(self, name, size, values=None):
        """
        Convention: use lower-case strings for indices        
        
        if no values are provided, we assume an integer index
        """
        self.indexSizes[name] = size
        if  values is None:
            self.indexValues[name] = None
        else:
            self.indexValues[name] = list(values)
        
    def registerTensor(self, name, indexTuples):
        """
        Make the manager aware of a specific tensor and the shape of its indices
        
        Only "input" tensors need to be registered; tensors computed by contraction are registered automatically when registerContraction() is called
        
        Convention: use Upper-Case letters only for tensor names
        
        (lower case i and t indicate inverses and transposes respectively, '_' indicates contraction)
        """
        self.tensorIndices[name] = indexTuples
        #reverse lookup for index name->numerical dimension
        self.tensorIndexPositions[name] = ({},{})
        pos = 0
        for l in range(2):
            for iname in indexTuples[l]:
                self.tensorIndexPosition[name][l][iname] = pos
                pos = pos + 1
                
        self.tensorData[name] = _np.zeros(self.getShape(indexTuples))
        tensor_shape =  self.getShape((lower, upper))
        self.tensorShape[name] = tensor_shape
        self.tensorShapeFlattened[name] = (_np.prod(tensor_shape[0]) , _np.prod(tensor_shape[1]))
        view = self._ntm.tensorData[tensorname].view()
        view.shape = self.tensorShapeFlattened[tensorname]
        self.tensorDataAsFlattened[name] = view


    def registerBasisTensor(self, basistensorname, index_name_tuples,  index_value_tuples):
        """        
        construct a orthonormal basis tensor by specifying 
        
        Usually they are used to add a tensor to a slice of another tensor via contraction
        
        registerBasisTensor('e_motioneffort', (('r'),('r')) (('motion',), ('effort',)), )
        registerBasisTensor('e_effort_0_effort_0', (('r', 'g'),('r', 'g')) (('effort',0), ('effort',0)), )
        """
        self.registerTensor(basistensorname, index_name_tuples)
        coord_pos = []
        for l in range(2):
            for value, index_name in zip(index_value_tuples[l], self.tensorIndices[basistensorname][l]):
                if self.indexValues[index_name] is None:
                    coord_pos.append(value) #default to natural numbers as index
                elif value is None:
                    coord_pos.append(value) #None selects everything,
                else:
                    coord_pos.append(self.indexValues[index_name].index(value))
        self.tensorData[name][tuple(coord_pos)] = 1.0


    def registerContraction(self, tensornameA, tensornameB, resultTensorName=None):
        """
        

        Tensor contraction contracts any index that occurs in the upper index tuple of one and in the lower index tuple of the other tensor. All other indices are preserved in the output tensor

        Some special behaviors:
            * Contrary to the popular Ricci notation rules, it is possible to have the same index name in the lower and upper index tuples of a tensor, because they are distinguishable by virtue of being upper and lower indices. Followig Ricci notation strictly would require us to rename either of the two conflicting indices before and after the tensor contraction, which is a purely notational roundabout effort
            
        
        """
 
        tensorIndices = self.tensorIndices
       
        tensorDotAxesTuples=([],[])
        contractedIndicesA = (set(), set())
        contractedIndicesB = (set(), set())
        
        offset_lower_left = len(tensorIndices[tensornameA][0])
        offset_lower_right = len(tensorIndices[tensornameB][0])

        for k in range(2): #check upper-lower or lower-upper pairs?
            for i_left, name in enumerate (tensorIndices[tensornameA][k]):
                for i_right, name2 in enumerate(tensorIndices[tensornameB][1-k]):
                    if name == name2:
                        tensorDotAxesTuples[0].append(    k  * offset_lower_left  + i_left ) 
                        tensorDotAxesTuples[1].append( (1-k) * offset_lower_right + i_right ) 
                        contractedIndicesA[k].add(name)
                        contractedIndicesB[1-k].add(name2)
        
        resultTensorIndicesLower = tuple(set(tensorIndices[tensornameA][1]) - contractedIndicesA[1] | set(tensorIndices[tensornameB][1]) - contractedIndicesB[1])
        resultTensorIndicesUpper = tuple(set(tensorIndices[tensornameA][0]) - contractedIndicesA[0] | set(tensorIndices[tensornameB][0]) - contractedIndicesB[0])
        
        if resultTensorName is None:
            resultTensorName = tensornameA + ':' + tensornameB
            
        if name in self.tensorIndices:
            raise Warning("tensor name is already registered")
        self.tensorIndices[resultTensorName] = (resultTensorIndicesUpper,resultTensorIndicesLower)
        self.tensorData[resultTensorName] = _np.zeros(self.getShape(self.tensorIndices[resultTensorName]))      
        self.registeredContractions[resultTensorName] = (self.tensorData[tensornameA], self.tensorData[tensornameB], tuple(tensorDotAxesTuples))
        self.update_order.append(resultTensorName)
        return


    def registerTranspose(self, tensorname):
        """
        register a "transpose" operation: lower all upper indices, and raise all lower indices
        """
        resultname ='({})^t'.format(tensorname)
        upper, lower = self.tensorIndices[tensorname]
        self.tensorIndices[resultname] = (lower, upper)
        axes = list(range(len(upper),len(lower)+len(upper)))+list(range(0, len(upper))) #lower and upper indices swap places
        self.tensorData[resultname] = _np.transpose(self.tensorData[tensorname], axes=axes) #we assume this is returns a view
        self.update_order.append(resultname)        
        return resultname

    def registerInverse(self, tensorname, regularization=1e-12):
        """
        register an "inverse" operation: matrix (pseudo-)inverse between all upper and all lower indices
        
        I.e. A : A^-1 = Kronecker delta
        
        """
        upper, lower = self.tensorIndices[tensorname]
        resultname ='({})^#'.format(tensorname)
        out_shape =  self.getShape((upper, lower))
        in_shape =  self.getShape((lower, upper))
        flattenedInShape = (_np.prod(in_shape[0]) , _np.prod(in_shape[1]))

        inAsFlattend = self.tensorData[tensorname].view()
        inAsFlattend.shape = flattenedInShape
        
        self.registeredInverses[resultname] = (inAsFlattend, out_shape, regularization)
        self.self.tensorData[tensorname] = _np.zeros(out_shape)
        self.update_order.append(resultname)        
        return resultname


    def registerAddition(self, A, B, resultname=None):
        """
        register an addition operation
        """
        if self.tensorIndices(B) != self.tensorIndices(A):
            raise ValueError("tensors must have exactly the same indices!")
        if resultname == None:
            resultname ='({0}+{1})'.format(A,B)
        self.tensorData[resultname] = _np.zeros(self.getShape(A))
        self.registeredAdditions[resultname] = (self.tensorData[A], self.tensorData[B])
        self.update_order.append(resultname)        
        return resultname        

    def registerSubtraction(self, A, B, resultname=None):
        """
        register a subtraction operation (A-B)
        """
        if self.tensorIndices(B) != self.tensorIndices(A):
            raise ValueError("tensors must have exactly the same indices!")
        if resultname == None:
            resultname ='({0}-{1})'.format(A,B)
        self.tensorData[resultname] = _np.zeros(shape) 
        self.registeredSubtractions[resultname] = (self.tensorData[A], self.tensorData[B])
        self.update_order.append(resultname)        
        return resultname        

    def getFlattened(self, tensorname):
            view = self._ntm.tensorData[tensorname].view()
            view.shape = self.tensorShapeFlattened[tensorname]

    def getSample(self, meanTensorName, covTensorName):
        """
        wrapper method for scipy's multivariate_normal()
        """
        return _np.random.multivariate_normal(self.tensorDataAsFlattened[meanTensorName], self.tensorDataAsFlattened[covTensorName])

    def getShape(self, upperLowerTuple):
        """
        return shape of array holding tensor coordinates given the tupple of upper and lower index tuples
        """
        return tuple([self.indexSizes[n] for n in upperLowerTuple[0]+upperLowerTuple[1]])


    def getIndexNames(tensorname):
        return self.tensorIndices[tensorname]  


    def getUpperIndexPosition(tensorname, indexname):
        return self.tensorIndices[tensorname][0].index(indexname)    

    def getLowerIndexPosition(tensorname, indexname):
        return self.tensorIndices[tensorname][1].index(indexname) + len(self.tensorIndices[tensorname][0])


    def getSize(indexname):
        return self.indexSizes[indexname]

    
    def setTensor(self, name, values):
        """
        Use this setter to avoid breaking internal stuff
        """
        if name not in self.tensorData:
            raise ValueError()
         _np.copyto(self.tensorData[name], values)

    def update(self, *args):
        """
        recompute the tensors using the registered operation yielding them
        
        if no arguments are given, recompute all registered operations, in the order they were registered
        """        
        if args == (): #if no names are given, iterate through all registered operations to update all tensors
            for tensorname in self.update_order:
                self.updateTensor(*self.update_order)
            return
        for tensorName in args:
            #recompute a single oepration:           
            if tensorName in self.registeredContractions:
                args = self.registeredContractions[tensorName]
                self.tensorData[tensorName]  = _np.tensordot(args[0], args[1], axes=args[2])
            elif resultName in self.registeredInverses: 
                resulttensor =  self.tensorData[tensorName]
                M, out_shape, regularizer = self.registeredInverses[resulttensor]            
                inverted = _np.linalg.pinv(M, rcond = regularizer) 
                inverted.shape = out_shape
                _np.copyto(  self.tensorData[tensorName], inverted) #unfortunately, we cannot specify an out array for the inverse
            elif resultName in self.registeredTransposes:                
                pass #nothing to do, we're using views for copy-free transpose
            else:
                raise Warning("tensor seems not to be computed by a registered operation")
            

        




