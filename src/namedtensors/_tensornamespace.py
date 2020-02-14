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
import copy as _copy


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
        self.tensorIndexPositionsAll = {}
        self.indexSizes = { }
        self.indexValues = { }
        self.tensorData = {}            # contains the actual array 
        self._withExternalArrays = set()
        self.tensorDataAsFlattened = {} # flattened view on the actual array
        self.tensorDataAsFlattenedDiagonal = {} # diagonal view on the flattened tensor
        self.tensorDataDiagonal = {} # diagonal view on the tensor
        self.tensorShape = {}
        self.tensorShapeFlattened = {}
        self.registeredScalars = {}
        self.registeredAdditions = {}
        self.registeredSubtractions = {}
        self.registeredScalarMultiplications = {}
        self.registeredElementwiseMultiplications = {}
        self.registeredContractions = {}
        self.registeredTransposes = {}
        self.registeredInverses = {}
        self.registeredExternalFunctions = {}
        self.registeredMeanOperators = {}
        self.registeredSliceOperations = {}
        self.registeredSumOperations = {}
        self._tensordot = _np.tensordot #which library to use
        self.update_order = []
        
        if ntm is not None:
            self.indexSizes = dict(ntm.indexSizes)
            self.indexValues = dict(ntm.indexValues)

    def copy(self):
        clone = _copy.deepcopy(self)
        #need to restore numpy views which get deep-copied accidentally:
        for name in self._withExternalArrays:
            clone.tensorData[name] = self.tensorData[name]
        clone.tensorDataAsFlattened = {}
        for name in clone.tensorData:
            view_flat = clone.tensorData[name].view()
            view_flat.shape = clone.tensorShapeFlattened[name]
            clone.tensorDataAsFlattened[name] = view_flat
        for name in clone.registeredTransposes:
            tensorname, axes  = clone.registeredTransposes[name]
            clone.tensorData[name] = _np.transpose(clone.tensorData[tensorname], axes=axes)
        return clone
        

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

    def registerTensor(self, name, indexTuples, external_array = None, initial_values='zeros'):
        """
        Make the manager aware of a specific tensor and the shape of its indices
        
        Only "input" tensors need to be registered; tensors computed by operations are registered automatically when registerContraction() is called
        
        Convention: use Upper-Case letters only for tensor names
        
        (lower case i and t indicate inverses and transposes respectively, '_' indicates contraction)
        """
        if name in self.tensorIndices:
            raise Warning("tensor name is already registered")
                   
        self._setIndexInfo(name, indexTuples) #add index info to precomputed data structures
                
        tensor_shape =  self.getShape(indexTuples)
        n_upper  = len(indexTuples[0])
        tensor_shape_flattened = (int(_np.prod(tensor_shape[:n_upper])) , int(_np.prod(tensor_shape[n_upper:])))  #int needed because prod converts empty tuples into float 1.0

        if external_array is None:
            self.tensorData[name] = _np.empty(tensor_shape)
        else:
            if external_array.shape != tensor_shape:
                raise ValueError(f'{external_array.shape} != {tensor_shape}')
            self.tensorData[name] = external_array  #set a reference to an external data
        
        self.tensorShape[name] = tensor_shape
        self.tensorShapeFlattened[name] = tensor_shape_flattened
        view_flat = self.tensorData[name].view()
        view_flat.shape = tensor_shape_flattened
        self.tensorDataAsFlattened[name] = view_flat

        if view_flat.shape[0] == view_flat.shape[1]: #for "square" tensors, construct a view on the diagonal
            self.tensorDataAsFlattenedDiagonal[name] = _np.einsum('ii->i',view_flat) # diagonal view on the flattened tensor
            n_upper = len(self.tensorIndices[name][0])
            self.tensorDataDiagonal[name] = self.tensorDataAsFlattenedDiagonal[name].reshape(self.tensorShape[name][:n_upper]) #un-flattened, i.e. tensor with only upper indices
        
        if initial_values == 'zeros':
            self.setTensor(name, 0.0)
        elif initial_values == 'identity' or 'kroneckerdelta':
            if any([u != self._flipTrailingUnderline(l) for u,l in zip(*self.tensorIndices[name])]) or len(self.tensorIndices[name][0]) != len(self.tensorIndices[name][1]):
                raise ValueError("upper and lower indices for initializing to identity / Kronecker delta must match exactly (up to trailing underlines)!")
            self.setTensorToIdentity(name)
        elif initial_values == 'ones':
            self.setTensor(name, 1.0)
        else:
            raise ValueError("value of initial_values argument not recognized")

    def _setIndexInfo(self, name, indexTuples):
        indexPositions = ({},{})
        indexPositionsAll = {}
        pos = 0
        for l in range(2):
            for iname in indexTuples[l]:
                indexPositions[l][iname] = pos
                indexPositionsAll[iname] = pos
                pos = pos + 1
        self.tensorIndices[name] = indexTuples
        self.tensorIndexPositions[name] = indexPositions        
        self.tensorIndexPositionsAll[name] = indexPositionsAll
        

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

    def renameIndices(self, A, indices_to_rename, result_name=None, inPlace=False):
        """
        Rename indices of a tensor
        
        indices_to_rename: dictionary of index names to rename
        
        inPlace=True modifies the tensor in place. Warning: this may be very confusing if done *after* using the tensor as operand!
        """
        for index in indices_to_rename:
            if self.indexSizes[index] != self.indexSizes[indices_to_rename[index]]:
                raise ValueError("Renamed indices must match in size! (attempted {}->{})".format(index,indices_to_rename[index]))
             
        #map all indices of the tensor given the map:
        renamed_tuples=[]
        for tup in self.tensorIndices[A]:
            renamed_tuple = []
            for index in tup:
                if index in indices_to_rename:
                    renamed_tuple.append(indices_to_rename[index])
                else:
                    renamed_tuple.append(index)
            renamed_tuples.append(tuple(renamed_tuple))
        renamed_tuples = tuple(renamed_tuples)

        if inPlace:
            self._setIndexInfo(A, renamed_tuples) 
        else:        
            if result_name is None:
                result_name = "renamed(A)"
            self.registerTensor(result_name, renamed_tuples, external_array=self.tensorData[A])


    def registerElementwiseMultiplication(self, A, B, result_name=None, out_array=None):
        
        if result_name is None:
            result_name = A + '*' + B
            
        tuplesA = self.tensorIndices[A]
        tuplesB = self.tensorIndices[B]
        if tuplesA != tuplesB:
            try:
                B_permuter = [ tuplesB[0].index(name) for name in tuplesA[0] ] + [ len(tuplesA[0])+tuplesB[1].index(name) for name in tuplesA[1] ]
            except ValueError:
                raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(self.tensorIndices[A], self.tensorIndices[B]) )
        else:
            B_permuter = None
    
        self.registerTensor(result_name, self.tensorIndices[A], external_array=out_array)        
        self.registeredElementwiseMultiplications[result_name] = (A, B, B_permuter)
        self.update_order.append(result_name)
            
            

    def registerScalarMultiplication(self, A, scalar, result_name=None, out_array=None):
        
        if result_name is None:
            result_name = tensornameA + '*' + scalar
    
    
        if _np.isreal(scalar):
            pass
        elif self.tensorIndices[scalar] == ((),()):
            pass
        else:
            raise ValueError("scalar argument needs to be a (0,0) tensor!")
    
        self.registerTensor(result_name, self.tensorIndices[A], external_array=out_array)        
        self.registeredScalarMultiplications[result_name] = (A, scalar)
        self.update_order.append(result_name)
    

    def registerContraction(self, tensornameA, tensornameB, result_name=None, out_array=None, flip_underlines=False, align_result_to=None):
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
        resultTensorIndicesUpperFromA = list(tensorIndices[tensornameA][0])
        for i in contractedIndicesA[0]:
            resultTensorIndicesUpperFromA.remove(i)
        resultTensorIndicesUpperFromB = list(tensorIndices[tensornameB][0])
        for i in contractedIndicesB[0]:
            resultTensorIndicesUpperFromB.remove(i)

        if flip_underlines:
            resultTensorIndicesUpperFromA = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesUpperFromA])
            resultTensorIndicesLowerFromA = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesLowerFromA])
            resultTensorIndicesUpperFromB = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesUpperFromB])
            resultTensorIndicesLowerFromB = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesLowerFromB])

        #compute the permutation for making tensordot results conformant to our upper-lower index ordering convention, and/or to align to requested index order
        if align_result_to == None:
            align_result_to = (tuple( resultTensorIndicesUpperFromA + resultTensorIndicesUpperFromB), tuple(resultTensorIndicesLowerFromA + resultTensorIndicesLowerFromB))

        indexorder_tensordot = resultTensorIndicesUpperFromA + resultTensorIndicesLowerFromA + resultTensorIndicesUpperFromB + resultTensorIndicesLowerFromB

        permuter = []
        for ul in range(2):
            for index_name in align_result_to[ul]:
                permuter.append(indexorder_tensordot.index(index_name))

        if result_name is None:
            result_name = tensornameA + ':' + tensornameB
        
        
        self.registerTensor(result_name, align_result_to, external_array=out_array)
        self.registeredContractions[result_name] = (tensornameA, tensornameB, tuple(tensorDotAxesTuples), permuter)
        self.update_order.append(result_name)
        return

    def registerTranspose(self, tensorname, result_name=None, flip_underlines=True):
        """
        register a "transpose" operation: lower all upper indices, and raise all lower indices
        """
        if result_name == None:
            result_name ='({})^T'.format(tensorname)
        upper, lower = self.tensorIndices[tensorname]
        
             
        axes = list(range(len(upper),len(lower)+len(upper)))+list(range(0, len(upper))) #lower and upper indices swap places        
        view = _np.transpose(self.tensorData[tensorname], axes=axes) #we assume this is returns a view - replaces the array in tensorData

        if flip_underlines:
            result_upper = tuple([self._flipTrailingUnderline(n) for n in lower])
            result_lower = tuple([self._flipTrailingUnderline(n) for n in upper])
        else:
            result_upper,result_lower = lower, upper

        self.registerTensor(result_name, (result_upper,result_lower), external_array=view)
        #no need to save operand parameters for copmutation as we use views
        self.registeredTransposes[result_name] = (tensorname, axes)
        self.update_order.append(result_name)        
        return result_name

    def registerInverse(self, tensorname, result_name = None, side='left', regularization=1e-9, flip_underlines=True, out_array=None):
        """
        register an "inverse" operation: matrix (pseudo-)inverse between all upper and all lower indices
        
        Uses the Moore-Penrose pseudoinverse internally
        
        I.e. A : A^-1 = Kronecker delta
        
        """
        if side == 'upper': side='left'  #synonymous
        if side == 'lower': side='right' #synonymous
        
        upper, lower = self.tensorIndices[tensorname]
        result_upper,result_lower = lower, upper
        
        if flip_underlines: 
            if side=='left': #for left-sided ("upper-sided"?) inverse, only the inverse's upper indices's underlines are flipped
                result_upper = tuple([self._flipTrailingUnderline(n) for n in result_upper])
            if side=='right': #vice versa for right-sided inverse
                result_lower = tuple([self._flipTrailingUnderline(n) for n in result_lower])

        if result_name is None:
            result_name ='({})^#'.format(tensorname)
        
        self.registerTensor(result_name, (result_upper, result_lower), external_array=out_array)
        self.registeredInverses[result_name] = (tensorname, side, regularization)
        self.update_order.append(result_name)        
        return result_name


    def registerAddition(self, A, B, result_name=None, out_array=None, flip_underlines=False, align_result_to=None):
        """
        register an addition operation
        """
        tuplesA = self.tensorIndices[A]
        tuplesB = self.tensorIndices[B]
        A_permuter = None
        B_permuter = None
        if align_result_to == None:
            align_result_to = tuplesA

        views = []
        for name in (A,B):
            index_tuples = self.tensorIndices[name]
            if align_result_to == index_tuples:
                view = self.tensorData[name].view()
            else:                        
                try:
                    permuter = [ index_tuples[0].index(name) for name in align_result_to[0] ] + [ len(align_result_to[0])+index_tuples[1].index(name) for name in align_result_to[1] ]
                except ValueError:
                    raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(align_result_to, self.tensorIndices[name]) )
                view = _np.transpose(self.tensorData[name], axes=permuter)
            views.append(view)


        if result_name == None:
            result_name ='({0}+{1})'.format(A,B)

        if flip_underlines:
            result_upper = tuple([self._flipTrailingUnderline(n) for n in self.tensorIndices[A][0]])
            result_lower = tuple([self._flipTrailingUnderline(n) for n in self.tensorIndices[A][1]])
        else: 
            result_upper, result_lower = self.tensorIndices[A]
        
        self.registerTensor(result_name, align_result_to, external_array=out_array)
        self.registeredAdditions[result_name] = (A,B, views)
        self.update_order.append(result_name)        
        return result_name        

    def registerSubtraction(self, A, B, *, result_name=None, out_array=None, flip_underlines=False):
        """
        register a subtraction operation (A-B)
        """
        tuplesA = self.tensorIndices[A]
        tuplesB = self.tensorIndices[B]
        if tuplesA != tuplesB:
            try:
                B_permuter = [ tuplesB[0].index(name) for name in tuplesA[0] ] + [ len(tuplesA[0])+tuplesB[1].index(name) for name in tuplesA[1] ]
            except ValueError:
                raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(self.tensorIndices[A], self.tensorIndices[B]) )
        else:
            B_permuter = None
            
        if result_name == None:
            result_name ='({0}-{1})'.format(A,B)

        if flip_underlines:
            result_upper = tuple([self._flipTrailingUnderline(n) for n in self.tensorIndices[A][0]])
            result_lower = tuple([self._flipTrailingUnderline(n) for n in self.tensorIndices[A][1]])
        else: 
            result_upper, result_lower = self.tensorIndices[A]
            
        self.registerTensor(result_name, (result_upper,result_lower), external_array=out_array)            
        self.registeredSubtractions[result_name] = (A,B, B_permuter)
        self.update_order.append(result_name)        
        return result_name        

    def registerExternalFunction(self, func,  in_tensor_names, out_tensor_names, out_tensor_indices, result_name=None):
        if result_name == None:
            result_name = ",".join(out_tensor_names)
        for name, indices in zip(out_tensor_names, out_tensor_indices):
            self.registerTensor(name, indices)
        self.registeredExternalFunctions[result_name] = (func,  in_tensor_names, out_tensor_names)
        self.update_order.append(result_name)


    def registerMean(self, A, index_to_sum, *, result_name=None, out_array=None):
        """
        computes 1^i:A^i * (1^i:(1^i)^T), i.e. the mean across the mentioned index
        """
        if result_name == None:
            result_name = "mean_{}({})".format(index_to_sum,A)
        upper, lower = self.tensorIndexPositions[A]
        result_upper = upper.copy()
        result_lower = lower.copy()
        if index_to_sum in upper:
            dim = upper[index_to_sum]
            result_upper.pop(index_to_sum)
        elif index_to_sum in lower:
            dim = lower[index_to_sum]
            result_lower.pop(index_to_sum)
        else:
            raise ValueError("index to sum is not in tensor")

        self.registerTensor(result_name, (result_upper,result_lower))
        self.registeredMeanOperators[result_name] = (A, dim)
        self.update_order.append(result_name)


    def registerSlice(self, A, sliced_indices_values, result_name=None):
        """
        Map the slice of a tensor to a new tensor name
        """
        if result_name == None:
            result_name = "slice({})".format(A)
        slicedef, sliced_indextuples = self.makeSliceDef(A, sliced_indices_values)
        self.registerTensor(result_name, sliced_indextuples, external_array=self.tensorData[A][slicedef])
        #no need to save operand parameters for copmutation as we use views
        self.registeredSliceOperations[result_name] = (A, slicedef, sliced_indextuples)
        self.update_order.append(result_name)        
        return result_name


    def registerSum(self, *args, result_name=None):
        """
        sum tensors into a single tensor
        """
        if result_name == None:
            result_name = "sum({})".format(','.join( args))

        tuplesReference = self.tensorIndices[args[0]] #everything is coerced into this order
        views = [ self._alignDimensions(tuplesReference, self.tensorIndices[name],  self.tensorData[name]) for name in args ]
        
        self.registerTensor(result_name, tuplesReference)
        self.registeredSumOperations[result_name] = (args, views)
        self.update_order.append(result_name)        


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

    def _flipTrailingUnderline(self, name):
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


    
    def setTensor(self, name, values, arrayIndices=None):
        """
        Use this setter to avoid breaking internal stuff
        """
        if name not in self.tensorData:
            raise ValueError()
        if values is None:
            return
        if arrayIndices is not None:
            values = self._alignDimensions(self.tensorIndices[name], arrayIndices, values)
        _np.copyto(self.tensorData[name], values)


#    def setTensorSlice(self, name, values, sliced_indices, value_indexTuples):
#        """
#        write to a certain slice of the tensor
#        
#        Warnin: you need to make sure manually that index order is correct
#        """
#        if name not in self.tensorData:
#            raise ValueError()
#        if values is None:
#            return
#        slicedef = [slice(None)]* self.tensorData[name].ndim
#        for index_name in sliced_indices:
#            axis = self.tensorIndexPositionsAll[name][index_name]
#            slicedef[axis] = sliced_indices[index_name]
#        _np.copyto(self.tensorData[name][tuple(slicedef)], values)


    def setTensorSlice(self, name, sliced_indices_values, slice_name, slice_namespace=None):
        """
        write to a certain slice of the tensor
        
        name: Tensor to set slice offset
        
        slice_name: tensor to get data from
        
        sliced_indices_values: values for the indices being sliced
        
        If an index is in name but neither in slice_name nor in sliced_index_values, then we broadcast values across this index
        
        """
        if name not in self.tensorData:
            raise ValueError()
        
        if slice_namespace == None:
            slice_namespace = self
        
        slicedef, sliced_indextuples = self.makeSliceDef(name, sliced_indices_values)

        #set the specified slice of name to values from slice_name
        slicedtensordata = self.tensorData[name][tuple(slicedef)]
        values_aligned = self._alignDimensions(sliced_indextuples, slice_namespace.tensorIndices[slice_name], slice_namespace.tensorData[slice_name])        
        _np.copyto(slicedtensordata, values_aligned)


    def makeSliceDef(self, name, sliced_indices_values):
        """
        create numpy slice definition for accessing slices of a tensor
        
        Also returns the index tuples of the resulting view        
        """
        wildcard = slice(None)
        slicedef = []
        sliced_indextuples = []
        for ul in range(2):
            indices = []
            for index_name in self.tensorIndexPositions[name][ul]:
                if index_name in sliced_indices_values:  #this index is being sliced:
                    axis = self.tensorIndexPositionsAll[name][index_name]
                    label = sliced_indices_values[index_name]
                    if self.indexValues[index_name] == None:   #plain numeric index                 
                        slicedef.append(label)
                    else:
                        slicedef.append(self.indexValues[index_name].index(label))
                else:
                    slicedef.append(wildcard)  
                    indices.append(index_name)
            sliced_indextuples.append(tuple(indices))
        return tuple(slicedef), tuple(sliced_indextuples) 
        

    def addToTensor(self, name, values, arrayIndices=None):
        """
        Use this setter to avoid breaking internal stuff
        """
        if name not in self.tensorData:
            raise ValueError()
        if values is None:
            return
        if arrayIndices is not None:
            values = self._alignDimensions(self.tensorIndices[name], arrayIndices, values)
        _np.add(self.tensorData[name], values, out=self.tensorData[name])


    def _alignDimensions(self, tuplesA, tuplesB, arrayB):
        """
        returns a view where arrayB dimensions are ordered according to tuplesA
        """
        tuplesA_indices = tuple(tuplesA[0]) + tuple(tuplesA[1])
        tuplesB_indices = tuple(tuplesB[0]) + tuple(tuplesB[1])
        try:
            permuter = [ tuplesB_indices.index(name) for name in tuplesA_indices ]
        except ValueError as e:
            raise ValueError("Index names don't match: {} vs. {}".format(tuplesA, tuplesB))
        return _np.transpose(arrayB, axes=permuter)
    


    def setTensorFromFlattened(self, name, values):
        """
        set tensor data from a matrix corresponding to tensor when flattened
        
        Use this setter to avoid breaking internal stuff
        """
        if name not in self.tensorData:
            raise ValueError()
        if values is not None:
            _np.copyto(self.tensorDataAsFlattened[name], values)


    def setTensorToIdentity(self, name, scale=1.0):
        """
        set a (p,p)-type tensor to the Kronecker delta
        
        Warning: make sure that the sizes and order of upper and lower indices match, i.e. that:
            (a,b),(a_,b_) -> delta(a,a_) : delta(b,b_)
        
        """
        row, column = self.tensorShapeFlattened[name]        
        if row != column:  #not exactly what we want to test
            raise ValueError("Cannot set identity if upper and lower indices don't match!")
        self.setTensor(name, 0.0)
        _np.fill_diagonal(self.tensorDataAsFlattened[name], scale)


    def update(self, *args, until=None):
        """
        recompute the tensors using the registered operation yielding them
        
        if no arguments are given, recompute all registered operations, in the order they were registered
        """        
        if until != None:
            args = self.update_order[:self.update_order.index(until)+1]
        elif args == (): #if no names are given, iterate through all registered operations to update all tensors
            args = self.update_order
        

        for result_name in args:
            #recompute a single oepration:
            A = ""
            B = ""  
            operation = ""
            B_permuter = ""  
            try:     
                if result_name in self.registeredContractions:
                    operation = "contract"
                    A,B,summing_pairs, reorder_upper_lower_indices = self.registeredContractions[result_name]
                    B_permuter = summing_pairs
                    td  = _np.tensordot(self.tensorData[A], self.tensorData[B], axes=summing_pairs)
                    _np.copyto(self.tensorData[result_name], _np.transpose(td, reorder_upper_lower_indices))
                    
                elif result_name in self.registeredInverses: 
                    operation = "invert"
                    A, side, regularizer = self.registeredInverses[result_name]
                    if side == 'left':
                        ATA = _np.dot(self.tensorDataAsFlattened[A].T,self.tensorDataAsFlattened[A])
                        ATA = ATA + _np.diag([regularizer]*ATA.shape[0])
                        ATAInv = _np.linalg.inv(ATA)
                        _np.dot(ATAInv, self.tensorDataAsFlattened[A].T, out=self.tensorDataAsFlattened[result_name])
                    elif side=='right': #right-sided pseudoinverse
                        AAT = _np.dot(self.tensorDataAsFlattened[A],self.tensorDataAsFlattened[A].T)
                        AAT = AAT + _np.diag([regularizer]*AAT.shape[0])
                        AATInv = _np.linalg.inv(AAT)
                        _np.dot(self.tensorDataAsFlattened[A].T, AATInv, out=self.tensorDataAsFlattened[result_name])
                    else:
                        raise ValueError()

                elif result_name in self.registeredInverses: 
                    operation = "invert_svd"
                    A, regularizer = self.registeredInverses[result_name]
                    inverted = _np.linalg.pinv( self.tensorDataAsFlattened[A], rcond = regularizer) 
                    inverted.shape = self.tensorShape[result_name]
                    _np.copyto(  self.tensorData[result_name], inverted) #unfortunately, we cannot specify an out array for the inverse
                    
                elif result_name in self.registeredTransposes:                
                    operation = "transpose"
                    pass #nothing to do, we're using views for copy-free transpose
                
                elif result_name in self.registeredAdditions:             
                    operation = "add"
                    A,B, views = self.registeredAdditions[result_name]
                    _np.add(views[0],views[1], out=self.tensorData[result_name])
                
                elif result_name in self.registeredSubtractions:                
                    operation = "subtract"
                    A,B, B_permuter = self.registeredSubtractions[result_name]
                    if B_permuter != None:
                        Bdata = _np.transpose(self.tensorData[B], axes=B_permuter) #precompute would be better, but so we avoid stale references
                    else:
                        Bdata = self.tensorData[B]
                    _np.subtract(self.tensorData[A], Bdata, out= self.tensorData[result_name])
                    
                elif result_name in self.registeredScalarMultiplications:  
                    operation = "scalar multiplication"
                    A,scalar = self.registeredScalarMultiplications[result_name]
                    if not _np.isreal(scalar):
                        scalar = self.tensorData[scalar]
                    _np.multiply(self.tensorData[A], scalar, out=self.tensorData[result_name])   #uses implicit broadcasting for scalar
                             
                elif result_name in self.registeredElementwiseMultiplications:  
                    operation = "scalar multiplication"
                    A,B,B_permuter = self.registeredElementwiseMultiplications[result_name]                  
                    if B_permuter != None:
                        Bdata = _np.transpose(self.tensorData[B], axes=B_permuter) #precompute would be better, but so we avoid stale references
                    else:
                        Bdata = self.tensorData[B]
                    _np.multiply(self.tensorData[A], Bdata, out=self.tensorData[result_name])            
                    
                elif result_name in self.registeredExternalFunctions:
                    operation = "function call"
                    func, in_tensor_names, out_tensor_names = self.registeredExternalFunctions[result_name]
                    func(self, in_tensor_names, out_tensor_names) #call external function with signature func(tns, in_tensor_tuple, out_tensor_tuple)
                    
                elif result_name in self.registeredMeanOperators:
                    operation = "mean"
                    A, dim = self.registeredMeanOperators[result_name]
                    _numpy.mean(self.tensorData['A'], axis=dim, out=self.tensorData[result_name], keepdims=False)

                elif result_name in self.registeredSliceOperations:
                    operation = "slice"
                    pass #nothing to do, we're using views for copy-free slice

                elif result_name in self.registeredSumOperations:
                    operation = "sum"
                    names, dataviews = self.registeredSumOperations[result_name]  
                    Z = self.tensorData[result_name]
                    Z[...]=0.0
                    for dataview in dataviews:
                        Z += dataview
                    
                else:
                    operation = "unknown"
                    raise Warning("tensor {} seems not to be computed by a registered operation".format(result_name))
            except Exception as e:
                
                print("Exception when computing {}={}({} , {})".format(result_name,operation,A,B))
                if A != "":
                    print("Details for {}: {}   {}".format(A, self.tensorIndices[A], self.tensorData[A].shape))
                if B != "":
                    print("Details for {}: {}   {}  {}".format(B, self.tensorIndices[B], self.tensorData[B].shape, B_permuter))
                if result_name != "":
                        for name in result_name.split(','):
                            print("Details for {}: {}   {}".format(name, self.tensorIndices[name], self.tensorShape[name]))
                raise e
                            

        




