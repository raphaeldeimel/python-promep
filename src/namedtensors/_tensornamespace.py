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
import collections as _collections

#Attributes of indices, accessible through the namespace['indexname'].attribute interface
indexDescription = _collections.namedtuple('IndexDescription', 'size values is_ordered_integers values_position')
#Attributes of tensors, accessible through the namespace['tensorname'].attribute interface
tensorDescription = _collections.namedtuple('TensorDescription', 'index_tuples indices_upper indices_lower ndim ndim_upper ndim_lower shape indices_position indices_upper_position indices_lower_position data data_diagonal shape_flat data_flat data_flat_diagonal data_are_external')


#Here you can change the dtype for all operations and data. Mostly for testing numeric stability with lesser resolution, or for saving memory
default_dtype = _np.float64

from scipy.linalg import lapack as _lapack
if default_dtype == _np.float32:
    def _inverse(A):
        return _lapack.spotri(A)[0]  #slower than float64, 40.8us  vs. 27.5us
else:
    def _inverse(A):
        return _np.linalg.inv(A)  #uses dgetri under the hood
#        return _lapack.dpotri(A)[0]  #twice as fast as _np.linalg.inv: 27.5us for a ^2x2x8_2x2x8 tensor vs. 65us, but not as stable for mixing


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
        self.registeredOperations = {}
        self._tensordot = _np.tensordot #which library to use
        self.update_order = []
        self._containeritems = {}
        self.index_names = set()
        self.tensor_names = []

        self.dtype  = default_dtype
        
        if ntm is not None:
            for name in ntm.index_names:
                if name.endswith('_'):
                    continue
                if ntm[name].is_ordered_integers:
                    self.registerIndex(name, ntm[name].size)
                else:
                    self.registerIndex(name, ntm[name].size, ntm[name].values)


        

    def registerIndex(self, name, size, values=None):
        """
        Convention: use lower-case strings for indices        
        
        if no values are provided, we assume an integer index
        """
        if name[-1] == '_':
            raise ValueError("trailing underscore is a special character for index names")
        if values == None:
            is_ordered_integers = True
            values = list(range(size))
        else:
            is_ordered_integers = False
            values = list(values)

        values_position  = {values[pos]:pos for pos in range(size) } #reverse lookup

        desc = indexDescription(size, values, is_ordered_integers, values_position)
        if name in self._containeritems:
            raise ValueError("Name {} already used: {}".format(name, self._containeritems[name]))
        name_ = name + '_'
        self._containeritems[name] = desc
        self._containeritems[name_] = desc
        self.index_names.add(name)
        self.index_names.add(name_)
        

        return name, name_


    def cloneIndex(self, clonefrom, newname):
        """
        Create a new index with identical size and values of thegiven index
        
        """
        if newname in self.index_names:
            return newname, newname+'_'
        else:
            if self[clonefrom].is_ordered_integers:
                name, name_ = self.registerIndex(newname, self[clonefrom].size)
            else:
                name, name_ = self.registerIndex(newname, self[clonefrom].size, self[clonefrom].values)
            return name, name_


    def registerTensor(self, name, description, external_array = None, initial_values='zeros'):
        """
        Make the manager aware of a specific tensor and the shape of its indices
        
        description: 
                Either a tuple of index tuples, e.g. (('a', 'b')('c',))
                
                Or a tensorDescription object, e.g. from another namespace
        
        external_array: if data are to be stored somewhere else, set the ndarray here
        
        initial_values: 'zero', 'ones', 'identity', 'keep'
        
        
        Only "input" tensors need to be registered; tensors computed by operations are registered automatically when registerContraction() is called
        
        Convention: use Upper-Case letters only for tensor names
        
        (lower case i and t indicate inverses and transposes respectively, '_' indicates contraction)
        
        
        """
        if name in self._containeritems:
            raise ValueError("Name already taken: {}".format(self._containeritems[name]))
           
        if description.__class__ == tensorDescription: #requested registering of a tensor from another namespace
            self._containeritems[name] = description
            self.tensor_names.append(name)
            return
        
        #else: tensor from scratch:
        indexTuples = description
            
        #catch a common mistake due to a pecularity of python syntax:
        if indexTuples[0].__class__ == str or indexTuples[1].__class__ == str:
            raise ValueError("Check the provided indexTuples: you provided a tuple of strings instead of a tuple of tuples.\nYou provided: {}\nHint: You may have forgotten a trailing colon, e.g.: (('alpha'),()) instead of (('alpha',),()).".format(indexTuples))
        
        indexTuples = (tuple(indexTuples[0]),tuple(indexTuples[1]))  #enforce uniformity of index definitions
        
        index2position_upper, index2position_lower, index2position = self._compute_index_reverse_lookup(indexTuples)

        tensor_shape =  tuple([self[n].size for n in indexTuples[0]] + [self[n].size for n in indexTuples[1]])
        
        n_upper  = len(indexTuples[0])
        tensor_shape_flattened = (int(_np.prod(tensor_shape[:n_upper])) , int(_np.prod(tensor_shape[n_upper:])))  #int needed because prod converts empty tuples into float 1.0

        if external_array is None:
            dataArray = _np.zeros(tensor_shape, dtype=self.dtype)
        else:
            if external_array.shape != tensor_shape:
                raise ValueError('{}!={}'.format(external_array.shape, tensor_shape))
            dataArray = external_array  #set a reference to an external data
        
        try:
            view_flat = dataArray.view()
            view_flat.shape = tensor_shape_flattened
            if view_flat.shape[0] == view_flat.shape[1]: #for "square" tensors, construct a view on the diagonal
                view_flat_diagonal = _np.einsum('ii->i',view_flat) # diagonal view on the flattened tensor
                view_diagonal = view_flat_diagonal.reshape(tensor_shape[:n_upper]) #un-flattened, i.e. tensor with only upper indices
            else:
                view_diagonal = None
                view_flat_diagonal = None

        except AttributeError:  #if we cannot create a flattened view, warn and continue
            print('"warning: could not create a flat view for' + name)
            view_diagonal = None
            view_flat_diagonal = None
            view_flat = None
            pass 
        
        indices_position, index2position_upper, index2position_lower = self._compute_index_reverse_lookup(indexTuples)
        if external_array is None:
            data_are_external = False
        else:
            data_are_external = True
            
        
        desc = tensorDescription(indexTuples, indexTuples[0], indexTuples[1], len(tensor_shape), len(indexTuples[0]), len(indexTuples[1]), tensor_shape, indices_position, index2position_upper, index2position_lower, dataArray, view_diagonal, tensor_shape_flattened, view_flat,  view_flat_diagonal, data_are_external)
        self._containeritems[name] = desc
        self.tensor_names.append(name)
        
        self.resetTensor(name,initial_values)


    def _compute_index_reverse_lookup(self, indexTuples):
        indexPositions = ({},{})
        indexPositionsAll = {}
        pos = 0
        for l in range(2):
            for iname in indexTuples[l]:
                indexPositions[l][iname] = pos
                indexPositionsAll[iname] = pos
                pos = pos + 1
        return indexPositionsAll, indexPositions[0],indexPositions[1]
        

    def registerBasisTensor(self, basistensorname, index_name_tuples,  index_value_tuples, ignoreLabels=False):
        """        
        construct a orthonormal basis tensor by specifying 
        
        Usually they are used to add a tensor to a slice of another tensor via contraction
        
        registerBasisTensor('e_motioneffort', (('r'),('r')) (('motion',), ('effort',)), )
        registerBasisTensor('e_effort_0_effort_0', (('r', 'g'),('r', 'g')) (('effort',0), ('effort',0)), )
        """
        self.registerTensor(basistensorname, index_name_tuples)
        coord_pos = []
        for l in range(2):
            for value, index_name in zip(index_value_tuples[l], self[basistensorname].index_tuples[l]):
                if value is None:
                    coord_pos.append(value) #None selects everything
                elif self[index_name].values is None or ignoreLabels:
                    coord_pos.append(value) #default to natural numbers as index
                else:  #do label-based lookup of position within data array
                    labels = self[index_name].values
                    labelpos = labels.index(value)
                    coord_pos.append(labelpos)
        self[basistensorname].data[tuple(coord_pos)] = 1.0
        return basistensorname

    def renameIndices(self, A, indices_to_rename, result_name=None, inPlace=False):
        """
        Rename indices of a tensor
        
        indices_to_rename: dictionary of index names to rename
        
        inPlace=True modifies the tensor in place. Warning: this may be very confusing if done *after* using the tensor as operand!
        """
        for index in indices_to_rename:
            if self[index].size != self[indices_to_rename[index]].size:
                raise ValueError("Renamed indices must match in size! (attempted {}->{})".format(index,indices_to_rename[index]))
             
        #map all indices of the tensor given the map:
        renamed_tuples=[]
        for tup in self[A].index_tuples:
            renamed_tuple = []
            for index in tup:
                if index in indices_to_rename:
                    renamed_tuple.append(indices_to_rename[index])
                else:
                    renamed_tuple.append(index)
            renamed_tuples.append(tuple(renamed_tuple))
        renamed_tuples = tuple(renamed_tuples)

        if inPlace:
            index2position, index2position_upper, index2position_lower = self._compute_index_reverse_lookup(renamed_tuples)        
            d = self._containeritems[A]
            d_renamed = tensorDescription(renamed_tuples, renamed_tuples[0], renamed_tuples[1], d[3], d[4], d[5], d[6], index2position,  index2position_upper, index2position_lower, d[10],d[11], d[12], d[13], d[14], d[15])
            self._containeritems[A] = d_renamed
        else:        
            if result_name is None:
                result_name = "renamed({})".format(A)
            self.registerTensor(result_name, renamed_tuples, external_array=self[A].data)
        return result_name


    def registerElementwiseMultiplication(self, A, B, result_name=None, out_array=None, initial_values_out_array='keep'):
        
        if result_name is None:
            result_name = A + '*' + B
            
        tuplesA = self[A].index_tuples
        tuplesB = self[B].index_tuples
        if tuplesA != tuplesB:
            try:
                B_permuter = [ tuplesB[0].index(name) for name in self[A].indices_upper ] + [ self[A].ndim_upper+self[B].indices_lower.index(name) for name in self[A].indices_lower ]
            except ValueError:
                raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(self[A].index_tuples, self[B].index_tuples) )
        else:
            B_permuter = None
    
        self.registerTensor(result_name, self[A].index_tuples, external_array=out_array,initial_values=initial_values_out_array)        
        self.registeredOperations[result_name] = ('elementwise_multiply', A, B, B_permuter)
        self.update_order.append(result_name)
        return result_name            
            

    def registerScalarMultiplication(self, A, scalar, result_name=None, out_array=None , initial_values_out_array='keep'):
        
        if result_name is None:
            result_name = "({}*{})".format(A,scalar)
    
        if _np.isreal(scalar):
            pass
        elif self[scalar].index_tuples == ((),()):
            pass
        else:
            raise ValueError("scalar argument needs to be a (0,0) tensor!")
    
        self.registerTensor(result_name, self[A].index_tuples, external_array=out_array, initial_values=initial_values_out_array)        
        self.registeredOperations[result_name] = ('scalar_multiply', A, scalar)
        self.update_order.append(result_name)
        return result_name

    def registerReset(self, A, initial_values='zeros'):
        """
        If executed, it sets coordinates back to the inital values
        """
        result_name = 'reset({})'.format(A)
        self.registeredOperations[result_name] = ('reset', A, initial_values)
        self.update_order.append(result_name)
        return result_name




    def registerContraction(self, tensornameA, tensornameB, result_name=None, out_array=None, initial_values_out_array='keep',  flip_underlines=False, align_result_to=None):
        """
        

        Tensor contraction contracts any index that occurs in the upper index tuple of one and in the lower index tuple of the other tensor. All other indices are preserved in the output tensor

        Some special behaviors:
            * Contrary to the popular Ricci notation rules, it is possible to have the same index name in the lower and upper index tuples of a tensor, because they are distinguishable by virtue of being upper and lower indices. Followig Ricci notation strictly would require us to rename either of the two conflicting indices before and after the tensor contraction, which is a purely notational roundabout effort
            
        
        """
       
        tensorDotAxesTuples=([],[])
        contractedIndicesA = (set(), set())
        contractedIndicesB = (set(), set())
        
        offset_lower_left = self[tensornameA].ndim_upper
        offset_lower_right = self[tensornameB].ndim_upper 

        for k in range(2): #check upper-lower or lower-upper pairs?
            for i_left, name in enumerate (self[tensornameA].index_tuples[k]):
                for i_right, name2 in enumerate(self[tensornameB].index_tuples[1-k]):
                    if name == name2:
                        tensorDotAxesTuples[0].append(    k  * offset_lower_left  + i_left ) 
                        tensorDotAxesTuples[1].append( (1-k) * offset_lower_right + i_right ) 
                        contractedIndicesA[k].add(name)
                        contractedIndicesB[1-k].add(name2)
        
        resultTensorIndicesLowerFromA = list(self[tensornameA].indices_lower)
        for i in contractedIndicesA[1]:
            resultTensorIndicesLowerFromA.remove(i)
        resultTensorIndicesLowerFromB = list(self[tensornameB].indices_lower)
        for i in contractedIndicesB[1]:
            resultTensorIndicesLowerFromB.remove(i)
        resultTensorIndicesUpperFromA = list(self[tensornameA].indices_upper)
        for i in contractedIndicesA[0]:
            resultTensorIndicesUpperFromA.remove(i)
        resultTensorIndicesUpperFromB = list(self[tensornameB].indices_upper)
        for i in contractedIndicesB[0]:
            resultTensorIndicesUpperFromB.remove(i)

        if flip_underlines:
            resultTensorIndicesUpperFromA = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesUpperFromA])
            resultTensorIndicesLowerFromA = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesLowerFromA])
            resultTensorIndicesUpperFromB = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesUpperFromB])
            resultTensorIndicesLowerFromB = tuple([self._flipTrailingUnderline(n) for n in resultTensorIndicesLowerFromB])
    
        #ensure Ricci notation rule:
        for idx in resultTensorIndicesUpperFromA:
            if idx in resultTensorIndicesUpperFromB:
                raise ValueError("Tensors both have the same upper index ({}) - forbidden by Ricci notation\n{}: {}\n{}: {}".format(idx, tensornameA, self[tensornameA].index_tuples,tensornameB, self[tensornameB].index_tuples))
        #ensure Ricci notation rule:
        for idx in resultTensorIndicesLowerFromA:
            if idx in resultTensorIndicesLowerFromB:
                raise ValueError("Tensors both have the same lower index ({}) - forbidden by Ricci notation\n{}: {}\n{}: {}".format(idx, tensornameA, self[tensornameA].index_tuples,tensornameB, self[tensornameB].index_tuples))

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
        
        
        self.registerTensor(result_name, align_result_to, external_array=out_array, initial_values=initial_values_out_array)
        self.registeredOperations[result_name] = ('contract', tensornameA, tensornameB, tuple(tensorDotAxesTuples), permuter)
        self.update_order.append(result_name)
        return result_name

    def registerTranspose(self, tensorname, result_name=None, flip_underlines=True):
        """
        register a "transpose" operation: lower all upper indices, and raise all lower indices
        """
        if result_name == None:
            result_name ='({})^T'.format(tensorname)
        upper, lower = self[tensorname].index_tuples
        
             
        axes = list(range(len(upper),len(lower)+len(upper)))+list(range(0, len(upper))) #lower and upper indices swap places        
        view = _np.transpose(self[tensorname].data, axes=axes) #we assume this is returns a view - replaces the array in data

        if flip_underlines:
            result_upper = tuple([self._flipTrailingUnderline(n) for n in lower])
            result_lower = tuple([self._flipTrailingUnderline(n) for n in upper])
        else:
            result_upper,result_lower = lower, upper

        self.registerTensor(result_name, (result_upper,result_lower), external_array=view)
        #no need to save operand parameters for copmutation as we use views
        self.registeredOperations[result_name] = ('transpose', tensorname, axes)
        self.update_order.append(result_name)        
        return result_name

    def registerInverse(self, tensorname, result_name = None, side='left', regularization=1e-10, flip_underlines=True, out_array=None, initial_values_out_array='keep'):
        """
        register an "inverse" operation: matrix (pseudo-)inverse between all upper and all lower indices
        
        Uses the Moore-Penrose pseudoinverse internally
        
        I.e. A : A^-1 = Kronecker delta
        
        """
        if side == 'upper': side='left'  #synonymous
        if side == 'lower': side='right' #synonymous
        
        upper, lower = self[tensorname].index_tuples
        result_upper,result_lower = lower, upper
        
        if flip_underlines: 
            if side=='left': #for left-sided ("upper-sided"?) inverse, only the inverse's upper indices's underlines are flipped
                result_upper = tuple([self._flipTrailingUnderline(n) for n in result_upper])
            if side=='right': #vice versa for right-sided inverse
                result_lower = tuple([self._flipTrailingUnderline(n) for n in result_lower])

        if result_name is None:
            result_name ='({})^#'.format(tensorname)
        
        self.registerTensor(result_name, (result_upper, result_lower), external_array=out_array, initial_values=initial_values_out_array)
        self.registeredOperations[result_name] = ('pseudoinvert', tensorname, side, regularization)
        self.update_order.append(result_name)        
        return result_name


    def registerAdditionToSlice(self, A, B, slice_indices={}):
        """
        register an addition operation
        
        accumulate: if true, add B to A (result_name, align_result_to, out_array, and flip_underlines becomes ineffective)
        """
        tuplesA = self[A].index_tuples
        B_permuter = None
        
        slicedefB, tuplesB = self.makeSliceDef(B, slice_indices)
 
        try:
            permuter = [ tuplesB[0].index(name) for name in tuplesA[0] ] + [ len(tuplesA[0])+tuplesB[1].index(name) for name in tuplesA[1] ]
        except ValueError:
            raise ValueError("tensors must have exactly the same indices! {} vs. {} + {}".format(tuplesA, tuplesB, list(slice_indices)) )

        slicedB = self[B].data[slicedefB]
        viewB = _np.transpose(slicedB, axes=permuter)


        txt = ''.join([idx+str(slice_indices[idx]) for idx in slice_indices])
        result_name = "accumulate({},{},{})".format(A,B, txt)
        self.registeredOperations[result_name] = ('add_to_slice', A,B,viewB, slice_indices)
        self.update_order.append(result_name)        
        return result_name        

    def registerAddition(self, A, B, result_name=None, out_array=None, initial_values_out_array='keep', flip_underlines=False, align_result_to=None, accumulate=False):
        """
        register an addition operation
        
        accumulate: if true, add B to A (result_name, align_result_to, out_array, and flip_underlines becomes ineffective)
        """
        tuplesA = self[A].index_tuples
        tuplesB = self[B].index_tuples
        A_permuter = None
        B_permuter = None
        if align_result_to == None or accumulate:
            align_result_to = tuplesA

        views = []
        for name in (A,B):
            index_tuples = self[name].index_tuples
            if align_result_to == index_tuples:
                view = self[name].data.view()
            else:                        
                try:
                    permuter = [ index_tuples[0].index(name_idx) for name_idx in align_result_to[0] ] + [ len(align_result_to[0])+index_tuples[1].index(name_idx) for name_idx in align_result_to[1] ]
                except ValueError:
                    raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(align_result_to, self[name].index_tuples) )
                view = _np.transpose(self[name].data, axes=permuter)
            views.append(view)


        if accumulate:
            result_name = A
            out_array = None
        elif result_name == None:
            result_name ='({0}+{1})'.format(A,B)

        if flip_underlines and not accumulate:
            result_upper = tuple([self._flipTrailingUnderline(n) for n in  self[A].indices_upper])
            result_lower = tuple([self._flipTrailingUnderline(n) for n in  self[A].indices_lower])
        else: 
            result_upper, result_lower = self[A].index_tuples
        
        self.registerTensor(result_name, align_result_to, external_array=out_array, initial_values=initial_values_out_array)
        self.registeredOperations[result_name] = ('add', A,B, views)
        self.update_order.append(result_name)        
        return result_name        

    def registerSubtraction(self, A, B, result_name=None, out_array=None, flip_underlines=False, initial_values_out_array='keep'):
        """
        register a subtraction operation (A-B)
        """
        tuplesA = self[A].index_tuples
        tuplesB = self[B].index_tuples
        if tuplesA != tuplesB:
            try:
                B_permuter = [ tuplesB[0].index(name) for name in tuplesA[0] ] + [ len(tuplesA[0])+tuplesB[1].index(name) for name in tuplesA[1] ]
            except ValueError:
                raise ValueError("tensors must have exactly the same indices! {} vs. {}".format(self[A].index_tuples, self[B].index_tuples) )
        else:
            B_permuter = None
            
        if result_name == None:
            result_name ='({0}-{1})'.format(A,B)

        if flip_underlines:
            result_upper = tuple([self._flipTrailingUnderline(n) for n in self[A].indices_upper])
            result_lower = tuple([self._flipTrailingUnderline(n) for n in self[A].indices_lower])
        else: 
            result_upper, result_lower = self[A].index_tuples
            
        self.registerTensor(result_name, (result_upper,result_lower), external_array=out_array, initial_values=initial_values_out_array)            
        self.registeredOperations[result_name] = ('subtract', A,B, B_permuter)
        self.update_order.append(result_name)        
        return result_name        

    def registerExternalFunction(self, func,  in_tensor_names, out_tensor_names, out_tensor_indices, result_name=None):
        if result_name == None:
            result_name = ",".join(out_tensor_names)
        for name, indices in zip(out_tensor_names, out_tensor_indices):
            self.registerTensor(name, indices)
        self.registeredOperations[result_name] = ('external_function', func,  in_tensor_names, out_tensor_names)
        self.update_order.append(result_name)
        return result_name

    def registerMean(self, A, index_to_sum, result_name=None, out_array=None, initial_values_out_array='keep'):
        """
        computes 1^i:A^i * (1^i:(1^i)^T), i.e. the mean across the mentioned index
        """
        if result_name == None:
            result_name = "mean_{}({})".format(index_to_sum,A)
        upper = self[A].indices_upper_position
        lower = self[A].indices_lower_position
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

        self.registerTensor(result_name, (result_upper,result_lower), out_array=out_array, initial_values=initial_values_out_array)
        self.registeredOperations[result_name] = ('mean', A, dim)
        self.update_order.append(result_name)
        return result_name

    def registerSlice(self, A, sliced_indices_values, result_name=None):
        """
        Map the slice of a tensor to a new tensor name
        """
        if result_name == None:
            result_name = "slice({})".format(A)
            if result_name in self.registeredSliceOperations:
                raise ValueError("a slice with the same name is already registered")
        slicedef, sliced_indextuples = self.makeSliceDef(A, sliced_indices_values)
        self.registerTensor(result_name, sliced_indextuples, external_array=self[A].data[slicedef], initial_values='keep')
        #no need to save operand parameters for copmutation as we use views
        self.registeredOperations[result_name] = ('slice', A, slicedef, sliced_indextuples)
        self.update_order.append(result_name)        
        return result_name


    def registerSum(self, sumTerms=None, result_name=None, sumcoordinates=False, out_array=None, initial_values_out_array='keep'):
        """
        sum multiple tensors into a single tensor
        
        if sumcoordinates is true, then also sum all coordinates of a tensor
        """
        if not isinstance(sumTerms, list):
            sumTerms = [sumTerms]

        if result_name == None:
            result_name = "sum({})".format(','.join( sumTerms))

        if sumcoordinates:
            tuplesReference = ((),()) #result is a scalar
            views = [self[name].data for name in sumTerms]
        else:
            tuplesReference = self[sumTerms[0]].index_tuples #everything is coerced into this order
            views = [ self._alignDimensions(tuplesReference, self[name].index_tuples,  self[name].data) for name in sumTerms ]
        
        self.registerTensor(result_name, tuplesReference, external_array=out_array, initial_values=initial_values_out_array)
        self.registeredOperations[result_name] = ('sum', sumTerms, views, sumcoordinates)
        self.update_order.append(result_name)        
        return result_name


    def getFlattened(self, tensorname):
            view = self._ntm[tensorname].data.view()
            view.shape = self.tensorShapeFlattened[tensorname]

    def getSample(self, meanTensorName, covTensorName):
        """
        wrapper method for scipy's multivariate_normal()
        """
        return _np.random.multivariate_normal(self[meanTensorName].data_flat, self[covTensorName].data_flat)


    def _flipTrailingUnderline(self, name):
        if name.endswith('_'):
            return name[:-1]
        else:
            return name + '_'


    def makeTensorSliceView(self, name, sliced_indices_values):
        """
        Make a view on the specified slice of tensor coordinates.
        
        This is mostly intended for easily converting from/to representations which
        specify things individually
        
        By returning a view, data can be accessed quickly in loops
        """
        slicedef, indices  = self.makeSliceDef(name, sliced_indices_values)
        view = self[name].data[slicedef]
        return view, indices
        

    def resetTensor(self, name, initial_values):
        """
        Set the tensor data to its initial value pattern
        """
        if initial_values == 'zeros':
            self.setTensor(name, 0.0)
        elif initial_values == 'keep':
            return
        elif initial_values == 'identity' or 'kroneckerdelta':
            if len(self[name].index_tuples) == 2:
                for ui, li in zip(self[name].indices_upper, self[name].indices_lower):
                    if self[ui].size != self[li].size:
                        raise Warning("Upper/Lower Index size does not match: {} vs. {}. Setting identity is not sensible.".format(ul, li))
            self.setTensorToIdentity(name)
        elif initial_values == 'ones':
            self.setTensor(name, 1.0)
        else:
            raise ValueError("value of initial_values argument not recognized")

    
    def setTensor(self, name, values, arrayIndices=None):
        """
        Use this setter to avoid breaking internal stuff
        
        name: tensor to set
        
        values: tensor name, or tensorDescription or numpy array to set tensor from
        
        arrayIndices: you can provide this for numpy arrays to do an additional check / auto-align the array's dimensions 
        """
        if name not in self.tensor_names:
            raise ValueError()
        if values is None:
            return
        if values.__class__ == tensorDescription: #we got handed a tensorDescription tuple?
            A = values
            values = self._alignDimensions(self[name].index_tuples, A.index_tuples, A.data)
        elif values.__class__ == str:
            A = values
            if not A in self.tensor_names: #we got handed a name string?
                raise ValueError("Input tensor named {} does not exist in the namespace".format(A))
            values = self._alignDimensions(self[name].index_tuples, self[A].index_tuples, self[A].data)
        elif arrayIndices is not None: #try to treat argument as numpy array
                values = self._alignDimensions(self[name].index_tuples, arrayIndices, values)
        else:
            pass #assume its a numpy array
        self[name].data[...] = values


    def setTensorSlice(self, name, sliced_indices_values, slice_name, slice_namespace=None):
        """
        write to a certain slice of the tensor
        
        name: Tensor to set slice offset
        
        slice_name: tensor to get data from
        
        sliced_indices_values: values for the indices being sliced
        
        If an index is in name but neither in slice_name nor in sliced_index_values, then we broadcast values across this index
        
        """
        if name not in self.tensor_names:
            raise ValueError()
        
        if slice_namespace == None:
            slice_namespace = self
        
        slicedef, sliced_indextuples = self.makeSliceDef(name, sliced_indices_values)

        #set the specified slice of name to values from slice_name
        slicedtensordata = self[name].data[tuple(slicedef)]
        if _np.isreal(slice_name): #user obviously just wants to set all elements to a common value (e.g. 0)
            _np.copyto(slicedtensordata, slice_name)        
        else:
            values_aligned = self._alignDimensions(sliced_indextuples, slice_namespace[slice_name].index_tuples , slice_namespace[slice_name].data)        
            _np.copyto(slicedtensordata, values_aligned)


    def makeSliceDef(self, name, sliced_indices_values):
        """
        create numpy slice definition for accessing slices of a tensor
        
        Also returns the index tuples of the resulting view        
        """
        wildcard = slice(None)
        slicedef = []
        sliced_indextuples = []
        for index2pos in (self[name].indices_upper_position, self[name].indices_lower_position) :
            indices = []
            for index_name in index2pos:
                if index_name in sliced_indices_values:  #this index is being sliced:
                    axis = self[name].indices_position[index_name]
                    label = sliced_indices_values[index_name]
                    if self[index_name].is_ordered_integers:
                       pos = label
                    else:
                       pos = self[index_name].values_position[label]
                    slicedef.append(pos)
                else:
                    slicedef.append(wildcard)  
                    indices.append(index_name)
            sliced_indextuples.append(tuple(indices))
        if len(slicedef) == len(self[name].shape):        
            slicedef.append(Ellipsis)  #this trick nudges numpy into always returning a view on the value if the tensor is a scalar (zero indices), instead of returning the value itsel
        return tuple(slicedef), tuple(sliced_indextuples) 
        

    def addToTensor(self, name, values, arrayIndices=None):
        """
        Use this setter to avoid breaking internal stuff
        """
        if values.__class__ == tensorDescription:
            view = self._alignDimensions(self[name].index_tuples, values.index_tuples, values.data)
            
        if name not in self.tensor_names:
            raise ValueError()
        if values is None:
            return
        if arrayIndices is not None:
            view = self._alignDimensions(self[name].index_tuples, arrayIndices, values)
        else:
            view = values
            
        _np.add(self[name].data, view, out=self[name].data)


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
        if name not in self.tensor_names:
            raise ValueError()
        if values is not None:
            _np.copyto(self[name].data_flat, values)


    def setTensorToIdentity(self, name, scale=1.0):
        """
        set a (p,p)-type tensor to the Kronecker delta
        
        Warning: make sure that the sizes and order of upper and lower indices match, i.e. that:
            (a,b),(a_,b_) -> delta(a,a_) : delta(b,b_)
        
        """
        row, column = self[name].shape_flat
        if row != column:  #not exactly what we want to test
            raise ValueError("Cannot set identity if upper and lower indices don't match!")
        self.setTensor(name, 0.0)
        _np.fill_diagonal(self[name].data_flat, scale)


    def update(self, *args):
        """
        recompute the tensors using the registered operation yielding them
        
        if no arguments are given, recompute all registered operations, in the order they were registered
        """        
        if args == (): #if no names are given, iterate through all registered operations to update all tensors
            order = self.update_order
        elif len(args) == 1 and not args[0].__class__== str: #we've probably been given a single list / tuple
            order = args[0]
        else:
            order=args

        for result_name in order:
            #recompute a single oepration:
            A = ""
            B = ""  
            operation = ""
            B_permuter = ""  
            operation_name  = self.registeredOperations[result_name][0]
            args  = self.registeredOperations[result_name][1:]
            try:
                if operation_name == 'reset':
                    A,initial_values = args
                    self.resetTensor(A, initial_values)
                elif operation_name == 'contract':
                    A,B,summing_pairs, reorder_upper_lower_indices = args
                    B_permuter = summing_pairs
                    td  = _np.tensordot(self[A].data, self[B].data, axes=summing_pairs)
                    _np.copyto(self[result_name].data, _np.transpose(td, reorder_upper_lower_indices))
                    
                elif operation_name == 'pseudoinvert':
                    operation = "invert"
                    A, side, regularizer = args
                    if side == 'left':
                        ATA = _np.dot(self[A].data_flat.T,self[A].data_flat)
                        ATA = ATA + _np.diag([regularizer]*ATA.shape[0])
                        ATAInv = _inverse(ATA)  #float32 cholesky decomposition based inverse
#                        ATAInv = _np.linalg.inv(ATA)
#                        print(ATAInv, self[A].data_flat.T.dtype,self[result_name].data_flat.dtype )
                        _np.dot(ATAInv, self[A].data_flat.T, out=self[result_name].data_flat)
                    elif side=='right': #right-sided pseudoinverse
                        AAT = _np.dot(self[A].data_flat,self[A].data_flat.T)
                        AAT = AAT + _np.diag([regularizer]*AAT.shape[0])
                        AATInv = _np.linalg.inv(AAT)
                        _np.dot(self[A].data_flat.T, AATInv, out=self[result_name].data_flat)
                    else:
                        raise ValueError()
                        
#                elif operation_name == 'inverse': 
#                    A, regularizer = args
#                    inverted = _np.linalg.pinv( self[A].data_flat, rcond = regularizer) 
#                    inverted.shape = self.tensorShape[result_name]
#                    _np.copyto(  self[result_name].data, inverted) #unfortunately, we cannot specify an out array for the inverse
                    
                elif operation_name == 'transpose':                
                    pass #nothing to do, we're using views for copy-free transpose
                
                elif operation_name == 'add': 
                    A,B, views = args
                    _np.add(views[0],views[1], out=self[result_name].data)
                
                elif operation_name == 'subtract':
                    A,B, B_permuter = args
                    if B_permuter != None:
                        Bdata = _np.transpose(self[B].data, axes=B_permuter) #precompute would be better, but so we avoid stale references
                    else:
                        Bdata = self[B].data
                    _np.subtract(self[A].data, Bdata, out= self[result_name].data)
                    
                elif operation_name == 'scalar_multiply':
                    A,scalar = args
                    if not _np.isreal(scalar):
                        scalar = self[scalar].data
                    _np.multiply(self[A].data, scalar, out=self[result_name].data)   #uses implicit broadcasting for scalar
                             
                elif operation_name == 'elementwise_multiply':
                    A,B,B_permuter = args                  
                    if B_permuter != None:
                        Bdata = _np.transpose(self[B].data, axes=B_permuter) #precompute would be better, but so we avoid stale references
                    else:
                        Bdata = self[B].data
                    _np.multiply(self[A].data, Bdata, out=self[result_name].data)            
                    
                elif operation_name == 'external_function':
                    func, in_tensor_names, out_tensor_names = args
                    func(self, in_tensor_names, out_tensor_names) #call external function with signature func(tns, in_tensor_tuple, out_tensor_tuple)
                    
                elif operation_name == 'mean':
                    A, dim = args
                    _np.mean(self['A'].data, axis=dim, out=self[result_name].data, keepdims=False)

                elif operation_name == 'slice':
                    pass #nothing to do, we're using views for copy-free slice

                elif operation_name == 'sum':
                    names, dataviews, sumcoordinates = args 
                    Z = self[result_name].data
                    Z[...]=0.0
                    for dataview in dataviews:
                        if sumcoordinates:
                            Z += _np.sum(dataview)
                        else:                    
                            Z += dataview
                elif operation_name == 'add_to_slice':
                    A,B,viewB, slice_indices = args
                    self[A].data[...] += viewB
                else:
                    raise Warning("tensor {} seems not to be computed by a registered operation {}".format(result_name, operation_name))
            except Exception as e:
                
                print("Exception when computing {}={}({} , {})".format(result_name,operation_name,A,B))
                if A != "":
                    print("Details for {}: {}   {}".format(A, self[A].index_tuples, self[A].shape))
                if B != "":
                    print("Details for {}: {}   {}  {}".format(B, self[B].index_tuples, self[B].shape, B_permuter))
                if result_name != "":
                        for name in result_name.split(','):
                            print("Details for {}: {}   {}".format(name, self[name].index_tuples, self[name].shape))
                raise e

    def __repr__(self):
        text = "Indices:\n"
        for index in self.index_names:
            if not index.endswith('_'):
                if self[index].is_ordered_integers:
                    valuestring  =  " (integers)"
                else:
                    valuestring = ": "+"{}".format(self[index].values)
                text += "{0}, {0}_: Size {1}{2}\n".format(index, self[index].size, valuestring)
        text += "\nTensors:\n"
        for name in self.tensor_names:
            text += "{}: {}  /  {}\n".format(name, " ".join(self[name].indices_upper), " ".join(self[name].indices_lower))
        return text

    def print_equations(self):
        """
        Print all registered operations (in update order) as nicely formatted equations
        
        Intended for debugging        
        """
        width = 10
        for result_name in self.update_order:
            width = max(width, len(result_name))            
        formatstring  ="{:>"+str(width)+"s} = {} ( {} )"
        s = []
        for result_name in self.update_order:
            args = self.registeredOperations[result_name]            
            operator = args[0]
            argstrings = []
            for arg in args[1:]:
                if arg.__class__ == str:
                    argstrings.append(arg)
                elif _np.isscalar(arg):
                    argstrings.append(arg.__repr__())                    
                elif arg.__class__ == list or arg.__class__ == tuple:
                    for argelement in arg:
                        if argelement.__class__ == str:
                            argstrings.append(argelement)                    
            s.append(formatstring.format(result_name, operator, ', '.join(argstrings) ) )
        print_string = "\n".join(s)
        print(print_string)


    # [] semantics:
    def __getitem__(self, key):
        return self._containeritems[key]
    def __setitem__(self, key, value):
        self._containeritems[key] = value
    def __delitem__(self, item):
        raise NotImplementedError()
    def keys(self):
        return self._containeritems.keys()


    
