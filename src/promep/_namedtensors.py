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


class TensorNameSpace(object):
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

    def __init__(self, ntm = None):
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
        
        if ntm is not None:
            self.indexSizes = dict(ntm.indexSizes)
            self.indexValues = dict(ntm.indexValues)
    

    def registerIndex(self, name, size, values=None):
        """
        Convention: use lower-case strings for indices        
        
        if no values are provided, we assume an integer index
        """
        if name[-1] == '_':
            raise ValueError("trailing underscore is a special character for index names")
        name2 = name + '_'
        self.indexSizes[name] = size
        self.indexSizes[name2] = size
        if values == None:
            self.indexValues[name] = None
            self.indexValues[name2] = None            
        else:
            self.indexValues[name] = list(values)
            self.indexValues[name2] = list(values)

    def registerTensor(self, name, indexTuples, external_array = None):
        """
        Make the manager aware of a specific tensor and the shape of its indices
        
        Only "input" tensors need to be registered; tensors computed by operations are registered automatically when registerContraction() is called
        
        Convention: use Upper-Case letters only for tensor names
        
        (lower case i and t indicate inverses and transposes respectively, '_' indicates contraction)
        """
        if name in self.tensorIndices:
            raise Warning("tensor name is already registered")
                    
        #reverse lookup for index name->numerical dimension
        indexPositions = ({},{})
        pos = 0
        for l in range(2):
            for iname in indexTuples[l]:
                indexPositions[l][iname] = pos
                pos = pos + 1
                
        tensor_shape =  self.getShape(indexTuples)
        n_upper  = len(indexTuples[0])
        tensor_shape_flattened = (int(_np.prod(tensor_shape[:n_upper])) , int(_np.prod(tensor_shape[n_upper:])))  #int needed because prod converts empty tuples into float 1.0

        self.tensorIndices[name] = indexTuples
        self.tensorIndexPositions[name] = indexPositions        
        if external_array is None:
            self.tensorData[name] = _np.zeros(tensor_shape)
        else:
            if external_array.shape != tensor_shape:
                raise ValueError(f'{external_array.shape} != {tensor_shape}')
            self.tensorData[name] = external_array  #set a reference to an external data
        self.tensorShape[name] = tensor_shape
        self.tensorShapeFlattened[name] = tensor_shape_flattened
        view_flat = self.tensorData[name].view()
        view_flat.shape = tensor_shape_flattened
        self.tensorDataAsFlattened[name] = view_flat


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
        self.tensorData[basistensorname][tuple(coord_pos)] = 1.0


    def registerContraction(self, tensornameA, tensornameB, result_name=None):
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
        
        resultTensorIndicesLowerFromA = list(tensorIndices[tensornameA][1])
        for i in contractedIndicesA[1]:
            resultTensorIndicesLowerFromA.remove(i)
        resultTensorIndicesLowerFromB = list(tensorIndices[tensornameB][1])
        for i in contractedIndicesB[1]:
            resultTensorIndicesLowerFromB.remove(i)
        resultTensorIndicesLower = resultTensorIndicesLowerFromA + resultTensorIndicesLowerFromB
        
        resultTensorIndicesUpperFromA = list(tensorIndices[tensornameA][0])
        for i in contractedIndicesA[0]:
            resultTensorIndicesUpperFromA.remove(i)
        resultTensorIndicesUpperFromB = list(tensorIndices[tensornameB][0])
        for i in contractedIndicesB[0]:
            resultTensorIndicesUpperFromB.remove(i)
        resultTensorIndicesUpper = resultTensorIndicesUpperFromA + resultTensorIndicesUpperFromB
        
        #compute the permutation for making tensordot results conformant to our upper-lower index ordering convention
        nau = len(resultTensorIndicesUpperFromA)
        nau_nal = nau + len(resultTensorIndicesLowerFromA)
        nau_nal_nbu = nau_nal + len(resultTensorIndicesUpperFromB)
        nau_nal_nbu_nbl = nau_nal_nbu + len(resultTensorIndicesLowerFromB)
        reorder_upper_lower_indices = list(range(0,nau)) + list(range(nau_nal, nau_nal_nbu)) + list(range(nau, nau_nal)) + list(range(nau_nal_nbu,nau_nal_nbu_nbl)) 
                
        if result_name is None:
            result_name = tensornameA + ':' + tensornameB
        
        self.registerTensor(result_name, (resultTensorIndicesUpper,resultTensorIndicesLower))
        self.registeredContractions[result_name] = (tensornameA, tensornameB, tuple(tensorDotAxesTuples), reorder_upper_lower_indices)
        self.update_order.append(result_name)
        return


    def registerTranspose(self, tensorname, result_name=None, flip_names=True):
        """
        register a "transpose" operation: lower all upper indices, and raise all lower indices
        """
        if result_name == None:
            result_name ='({})^t'.format(tensorname)
        upper, lower = self.tensorIndices[tensorname]
        
             
        axes = list(range(len(upper),len(lower)+len(upper)))+list(range(0, len(upper))) #lower and upper indices swap places        
        view = _np.transpose(self.tensorData[tensorname], axes=axes) #we assume this is returns a view - replaces the array in tensorData

        if flip_names:
            result_upper = tuple([self._flipIndexName(n) for n in lower])
            result_lower = tuple([self._flipIndexName(n) for n in upper])
        else:
            result_upper,result_lower = lower, upper

        self.registerTensor(result_name, (result_upper,result_lower), external_array=view)
        #no need to save operand parameters for copmutation as we use views
        self.registeredTransposes[result_name] = None
        self.update_order.append(result_name)        
        return result_name

    def registerInverse(self, tensorname, regularization=1e-12, flip_names=True):
        """
        register an "inverse" operation: matrix (pseudo-)inverse between all upper and all lower indices
        
        I.e. A : A^-1 = Kronecker delta
        
        """
        upper, lower = self.tensorIndices[tensorname]
        result_name ='({})^#'.format(tensorname)
        in_shape =  self.getShape((upper, lower))
        flattenedInShape = (_np.prod(in_shape[0]) , _np.prod(in_shape[1]))

        inAsFlattend = self.tensorData[tensorname].view()
        inAsFlattend.shape = flattenedInShape
        
        if flip_names:
            result_upper = tuple([self._flipIndexName(n) for n in lower])
            result_lower = tuple([self._flipIndexName(n) for n in upper])
        else:
            result_upper,result_lower = lower, upper

        
        self.registeredInverses[result_name] = (tensorname, regularization)
        self.registerTensor(result_name, (result_upper, result_lower), external_array=view)
        self.update_order.append(result_name)        
        return result_name


    def registerAddition(self, A, B, result_name=None):
        """
        register an addition operation
        """
        tuplesA = self.tensorIndices[A]
        tuplesB = self.tensorIndices[B]
        if tuplesA != tuplesB:
            try:
                B_permuter = [ tuplesB[0].index(name) for name in tuplesA[0] ] + [ len(tuplesA[0])+tuplesB[1].index(name) for name in tuplesA[1] ]
            except ValueError:
                raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(self.tensorIndices[B], self.tensorIndices[A]) )
        else:
            B_permuter = None
    
        if result_name == None:
            result_name ='({0}+{1})'.format(A,B)
        
        self.registerTensor(result_name, tuplesA)
        self.registeredAdditions[result_name] = (A,B, B_permuter)
        self.update_order.append(result_name)        
        return result_name        

    def registerSubtraction(self, A, B, result_name=None):
        """
        register a subtraction operation (A-B)
        """
        tuplesA = self.tensorIndices[A]
        tuplesB = self.tensorIndices[B]
        if tuplesA != tuplesB:
            try:
                B_permuter = [ tuplesB[0].index(name) for name in tuplesA[0] ] + [ len(tuplesA[0])+tuplesB[1].index(name) for name in tuplesA[1] ]
            except ValueError:
                raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(self.tensorIndices[B], self.tensorIndices[A]) )
        else:
            B_permuter = None
            
        if result_name == None:
            result_name ='({0}-{1})'.format(A,B)
            
        self.registerTensor(result_name, self.tensorIndices[A])            
        self.registeredSubtractions[result_name] = (A,B, B_permuter)
        self.update_order.append(result_name)        
        return result_name        

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
        return tuple([self.indexSizes[n] for n in upperLowerTuple[0]] + [self.indexSizes[n] for n in upperLowerTuple[1]])

    def _flipIndexName(self, name):
        if name.endswith('_'):
            return name[:-1]
        else:
            return name + '_'

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
        if values is not None:
            _np.copyto(self.tensorData[name], values)

    def setTensorToIdentity(self, name):
        """
        set a (p,p)-type tensor to the Kronecker delta
        
        Warning: make sure that the sizes and order of upper and lower indices match, i.e. that:
            (a,b),(a_,b_) -> delta(a,a_) : delta(b,b_)
        
        """
        row, column = self.tensorShapeFlattened[name]        
        if row != column:  #not exactly what we want to test
            raise ValueError("Cannot set identity if upper and lower indices don't match!")
        self.setTensor(name, 0.0)
        _np.fill_diagonal(self.tensorDataAsFlattened[name], 1.0)


    def update(self, *args):
        """
        recompute the tensors using the registered operation yielding them
        
        if no arguments are given, recompute all registered operations, in the order they were registered
        """        
        if args == (): #if no names are given, iterate through all registered operations to update all tensors
            args = self.update_order

        for result_name in args:
            #recompute a single oepration:
            try:     
                A = ""
                B = ""  
                operation = ""
                B_permuter = ""  
                if result_name in self.registeredContractions:
                    operation = "contract"
                    A,B,summing_pairs, reorder_upper_lower_indices = self.registeredContractions[result_name]
                    B_permuter = summing_pairs
                    td  = _np.tensordot(self.tensorData[A], self.tensorData[B], axes=summing_pairs)
                    _np.copyto(self.tensorData[result_name], _np.transpose(td, reorder_upper_lower_indices))
                    
                elif result_name in self.registeredInverses: 
                    operation = "invert"
                    A, regularizer = self.registeredInverses[result_name]
                    inverted = _np.linalg.pinv( self.tensorDataAsFlattened[A], rcond = regularizer) 
                    inverted.shape = self.tensorShape[result_name]
                    _np.copyto(  self.tensorData[result_name], inverted) #unfortunately, we cannot specify an out array for the inverse
                    
                elif result_name in self.registeredTransposes:                
                    operation = "transpose"
                    pass #nothing to do, we're using views for copy-free transpose
                
                elif result_name in self.registeredAdditions:             
                    operation = "add"
                    A,B, B_permuter = self.registeredAdditions[result_name]
                    if B_permuter != None:
                        Bdata = _np.transpose(self.tensorData[B], axes=B_permuter) #precompute would be better, but so we avoid stale references
                    else:
                        Bdata = self.tensorData[B]
                    _np.add(self.tensorData[A], Bdata, out= self.tensorData[result_name])
                
                elif result_name in self.registeredSubtractions:                
                    operation = "subtract"
                    A,B, B_permuter = self.registeredSubtractions[result_name]
                    if B_permuter != None:
                        Bdata = _np.transpose(self.tensorData[B], axes=B_permuter) #precompute would be better, but so we avoid stale references
                    else:
                        Bdata = self.tensorData[B]
                    _np.subtract(self.tensorData[A], Bdata, out= self.tensorData[result_name])
                else:
                    operation = "unknown"
                    raise Warning("tensor {} seems not to be computed by a registered operation".format(result_name))
            except Exception as e:
                print("Exception when computing {}={}({} , {})".format(result_name,operation,A,B))
                print("Details for {}: {}   {}".format(A, self.tensorIndices[A], self.tensorData[A].shape))
                print("Details for {}: {}   {}  {}".format(B, self.tensorIndices[B], self.tensorData[B].shape, B_permuter))
                print("Details for {}: {}   {}".format(result_name, self.tensorIndices[result_name], self.tensorShape[result_name]))
                
                raise e
                            

        




