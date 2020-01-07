#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains the main code for representing ProMPs, visualizing them, and
also to sample trajectories from a ProMP

The algorithms are described in:
[1] A. Paraschos, C. Daniel, J. Peters, and G. Neumann,
“Using probabilistic movement primitives in robotics,” Autonomous Robots, pp. 1–23, 2017.

"""

import numpy as _np
import itertools as _it
#import time as _time

def dot(*args, **kwargs):
    """ tensor "dot product" of all arguments

        effectively, dot(A,B,C,...) == A :: B :: C...

        with :: being the tensor double contraction for rank 4 tensors
        and the matrix multiplication for rank 2 tensors

        args: tensors of either rank 4, 2,1 or 0 (scalar)

        kwargs:  shape: tuple that specifies the shapes (upper + lower index counts) of the tensors in args
                    i.e. ((3,2),(2,0)) means first arg is a (3,2)-tensor, second arg is a (2,0)-tensor

        you can think of this operator as performing the equivalent of a matrix dot product
        
        special case optimization: if the two inner indices of two rank-4 tensors are block-diagonal, 
            i.e. if (A[:,i,j,:]) == 0 for i !=j, you can speed up the computation by setting:
                dofIdentity = True
            This is usually true for covariance Tensors with equal number of dofs in parameter and joint space
    """    
    res = _np.asarray(args[0])
    #print( [a.shape for a in args])
    if "shapes" in kwargs:
        shapes = kwargs["shapes"]
        lshape = shapes[0]
        for r, rshape in zip(args[1:], shapes[1:]):
            r = _np.asarray(r)
            if len(res.shape) != lshape[0]+lshape[1]:
                raise ValueError("Specified shape does not fit to array size: {0} != {1} + {2}".format(len(res.shape), lshape[0], lshape[1]))
            if len(r.shape) != rshape[0]+rshape[1]:
                raise ValueError("Specified shape does not fit to array size: {0} != {1} + {2}".format(len(r.shape), rshape[0], rshape[1]))
            if lshape[1] == rshape[0]:
                    res = _np.tensordot(res, r, axes=lshape[1])
            else:
                raise ValueError("specified tensor shapes do not fit: {0} : {1}".format(lshape, rshape))
            lshape = (lshape[0], rshape[1])
    else:
        for a in args[1:]:
            a = _np.asarray(a)
            if res.ndim == 4 and a.ndim == 4: #(assume (2,2)-tensors)
                if u'dofIdentity' in kwargs and kwargs[u'dofIdentity']==True:   #don't compute j!=k entries, they are zero
                    for j in range(res.shape[1]):
                        res[:,j,:,j] = _np.einsum('il,ln', res[:,j,:,j] , a[:,j,:,j])
                else:
                    res = _np.einsum('ijkl,klmn->ijmn', res, a)
            elif res.ndim == 4 and a.ndim == 2:
                res = _np.einsum('ijkl,kl->ij', res, a)
            elif res.ndim == 2 and a.ndim == 4:
                res = _np.einsum('kl,klmn->mn', res, a)
            elif res.ndim == 2 and a.ndim == 2:
                res = _np.einsum('kl,kl', res, a)
            elif res.ndim == 1 and a.ndim == 1:
                res = _np.einsum('l,l', res, a)
            elif a.ndim == 0 or res.ndim == 0:
                res = res * a
    return res
    
def T(tensor, **kwargs):
    """
    Return the tensor with upper and lower indices lowered and raised respectively
    
    shape: shape of the tensor
        specify this for non (2,2) or (1,1) tensors
    """
    if "shape" in kwargs:
        shape = kwargs["shape"]
        uindices = list(range(shape[0]))
        lindices = list(range(shape[0],shape[0]+shape[1]))
        return _np.transpose(tensor, lindices+uindices)
    else:
        A = _np.asarray(tensor)
        if A.ndim == 4: #(assume a (2,2)-tensor)
            return tensor.transpose(2,3,0,1)
        else:
            return tensor.T


def eigenValues(tensor):
    if tensor.ndim == 4: #(assume a (2,2)-tensor)
        flat = tensor.view().reshape(tensor.shape[0]*tensor.shape[1],tensor.shape[0]*tensor.shape[1])
        eigenvalues = _np.linalg.eigvals(flat)
        eigenvalues.reshape(tensor.shape[0]*tensor.shape[1])
        return eigenvalues
    elif tensor.ndim == 2: #(1,1)-tensor a.k.a. matrix
        return _np.linalg.eigvals(tensor)
    else:
        ValueError()

def pinv(M, rowindices=None, regularization=1e-12):
    """
    return the pseudoinverse of a tensor

    a: tensor of shape (m x n x ... x m x n x...)

    rowindices: how many of the indices are row indices
                    if None, assume an even split between row and column indices

    regularization: value to add for ridge regularization

    returns tensor of shape (m x n x ... x m x n x...)

    you can think of this operator as performing the equivalent of a matrix inversion
    """
    if rowindices == None:
        rowindices = M.ndim // 2
        if M.ndim % 2 != 0:
            ValueError("Notice: dimension of array is not even, but no explicit number of row indices was supplied. This probably is an error")
    #compute pseudoinverse using a a flattened array:
    original_shape = M.shape
    new_shape = M.shape[rowindices:] + M.shape[:rowindices]
    rowsize = _np.prod(M.shape[:rowindices])
    M = M.reshape((rowsize, -1))
    #need to do pseudoinverse by hand as adding a ridge regularization term is not implemented in _np.pinv
    MTM = _np.dot(M.T,M)
    if not regularization is None:
        MTM = MTM + _np.diag([regularization]*MTM.shape[0])
    AInv = _np.dot(_np.linalg.inv(MTM), M.T)
    AInv.shape = new_shape
    return AInv


def I(m,n, lam=1.0):
    """
    return an Identity tensor of shape <m,n | m,n>

    lam: scalar to multiply the identity tensor with
    """
    T = _np.zeros((m,n,m,n)) #TODO: convert to setting shape of flat identity matrix
    for i,j in _it.product(range(m), range(n)):
        if _np.isscalar(lam):
            T[i,j,i,j] = lam
        else:
            T[i,j,i,j] = lam[i,j]
    return T

def getDiagView(tensor):
    """
    returns a view on the diagonal of a tensor with identical lower and upper indices
    
    The tensor has to be of shape (n,n)
    
    """
    if tensor.ndim == 2: #(1,1)-tensor a.k.a. matrix
        return _np.einsum('ii->i', tensor)
    elif tensor.ndim == 4: #(assume a (2,2)-tensor)
        return _np.einsum('ijij->ij', tensor)
    elif tensor.ndim == 6: #(assume a (3,3)-tensor)
        return _np.einsum('ijkijk->ijk', tensor)
    else:
        ValueError()


def addToRidge(T, lam):
    for i,j in _it.product(range(T.shape[0]), range(T.shape[1])):
        T[i,j,i,j] += lam
    return T


def asList(tensor):
    """
    correctly serialize a tensor into a flat list using C-style array semantics
    
    note: The otherwise obvious numpy.flat does not honor swapped/modified axes correctly
    """
    return tensor.reshape(-1).tolist()
    
def flattenCovarianceTensor(cov):
    """
    helper function to correctly flatten the covariance tensor from <supports, dofs | supports, dofs>
    into matrix ((supports*dofs), (supports*dofs))
    """
    supports,dofs,supports_b,dofs_b = cov.shape
    if dofs != dofs_b or supports != supports_b:
        raise ValueError
    covarianceMatrix = _np.empty((supports*dofs,supports*dofs))
    for i,j in _it.product(range(dofs),range(dofs_b)):
        covarianceMatrix[i*supports:(i+1)*supports, j*supports_b:(j+1)*supports_b] = cov[:,i, :,j] #TODO: replace by reshaping
    return covarianceMatrix


def foldCovarianceMatrix(covarianceMatrix, supports, dofs):
    """
    helper function to correctly fold a covariance matrix  of shape ((supports*dofs), (supports*dofs))
    into a 4th-order tensor of shape < supports, dofs | supports, dofs >
    """
    a,b = covarianceMatrix.shape
    if a!=b:
        raise ValueError()
    if dofs*supports != a:
        raise ValueError()
    covarianceTensor = _np.empty((supports,dofs, supports,dofs)) 
    for i,j in _it.product(range(dofs),range(dofs)): #TODO: replace by reshaping
        covarianceTensor[:,i, :,j] = covarianceMatrix[i*supports:(i+1)*supports, j*supports:(j+1)*supports] 
    return covarianceTensor


def getStdDeviationsOfStateCovarianceTensor(covarianceTensor):
    """
    extract the standard deviations from the state covariance tensor

    return an (a x b) matrix from an < a, b | a, b > tensor
    """
    derivCount, dofs, derivCount2,dofs2 = covarianceTensor.shape
    if dofs != dofs2 or derivCount != derivCount2:
        raise ValueError("shape of covariance tensor is wrong!")
    var = _np.empty((derivCount,dofs))
    for dof in range(dofs):
        for g in range(derivCount):
            a = covarianceTensor[g,dof, g, dof]
            if a < 0.0:
                RuntimeWarning("Warning: negative variance encountered ({0}), cliping to 0".format(a))
                var[g,dof] = 0.0
            else:    
                var[g,dof] = a
    sigma = _np.sqrt(var)
    return sigma


def makeCovarianceTensorUncorrelated(supports, dofs, sigmas):
    cov = _np.zeros((supports,dofs, supports,dofs))
    if _np.isscalar(sigmas):
        for dof in range(dofs):
            for s in range(supports):
                cov[s,dof, s,dof] = sigmas**2
    else:
        for dof in range(dofs):
            for s in range(supports):
                cov[s,dof, s,dof] = sigmas[s,dof]**2
    return cov


