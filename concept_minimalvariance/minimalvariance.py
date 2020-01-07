#!/bin/ipython

import numpy as _np



def func_ref_min(c1,c2):
    """
    desired function for 1D case
    """
    return _np.minimum(c1,c2)

def func_ref(c1,c2):
    """
    mixing using the weighted average

    used by ProMP papers  (Paraschos, Maeda)
    """
    k=0.5
    return (k*c1**-1+k*c2**-1)**-1

def func_ref_sum(c1,c2):
    """
    naive mixing using the weighted average

    used in a lot of setups for motion distributions

    pro: simple
    con: reduces variances for equal input variances
    """
    k=0.5
    return (k*c1+k*c2)


def minimal_variances_3(c1,c2,c3=1000000.0, factors=(1.,-2,3.)):
    """
    prototype of minimal-variance mixing method:
    """
    c_ = (c1**-1+c2**-1+c3**-1)
    return  ( factors[0]*c_  + factors[1] * ((c1+c2)**-1 + (c1+c3)**-1 + (c2+c3)**-1  ) + factors[2] * (c1+c2+c3)**-1 )**-1



def minimal_variances_deimel(c1,c2, alpha1=1.0, alpha2=1.0, iterations=10):
    """
    prototype of minimal-variance mixing method:
    """
    #set up inverse to start with
    inv_c = 2*(c1+c2)**-1
    sum_alpha = alpha1 + alpha2
    sum_alpha2 = alpha1**2 + alpha2**2
    for i in range(iterations): 
        #iterate on inverse of mixture
        inv_c1_weighted = inv_c * (sum_alpha2 * alpha1 * inv_c * c1**2 + sum_alpha * (1.-alpha1))**-1
        inv_c2_weighted = inv_c * (sum_alpha2 * alpha2 * inv_c * c2**2 + sum_alpha * (1.-alpha2))**-1 
        inv_c = (inv_c1_weighted + inv_c2_weighted)
    
    return inv_c**-0.5


def mixing_deimel(c1,c2,c3=1000000.0):
    """
    prototype of minimal-variance mixing method:
    """
    #set up inverse to start with
    inv_c = _np.linalg.inv(c1+c2)
    sum_alpha = 2 #sum of activations
    sum_alpha2 = 2 #sum of squares of activations
    for i in range(30): 
        #iterate on inverse of mixture
        inv_c1_weighted = inv_c @ _np.linalg.inv(sum_alpha2 * inv_c @ c1 + 0 * sum_alpha) 
        inv_c2_weighted = inv_c @ _np.linalg.inv(sum_alpha2 * inv_c @ c2 + 0 * sum_alpha) 
        inv_c = inv_c2_weighted + inv_c2_weighted
    
    return  _np.linalg.inv(inv_c)



def minimal_variances(covariances, inv_func=_np.linalg.inv):
    """
    function to combine covariance matrices so that in the limit, minimal variances take precedence

    In its essence the function computes the product of all input distributions, and then adds correction terms to account for low variances that where "counted twice" when directions of low variance in input distributions align, then adds terms to account for over-corrected low variances when three input distributions align, and so forth.


        Example for three covariance matrices:
        S**-1 =  1 * (S1**-1 + S2**-1 +S3**-1 )
                -2 * ( (S1+S2)**-1 + (S1+S3)**-1 + (S2+S3)**-1 )
                +3 * ( (S1+S2+S3)**-1 )

    The mixing function's weighing factors are completely determined by the limit cases in the 1D case:
        func([lo, lo,lo,lo ] ) = lo
        func([hi, hi,hi,hi ] ) = hi
        "reproduce variance exactly if input distributions agree"
        (from a probabilistic point of view: assume maximum correlation between inputs)

        func([lo, hi,hi,hi ] ) = lo
        func([hi, lo,hi,hi ] ) = lo
        func([hi, hi,lo,hi ] ) = lo
        func([hi, hi,hi,lo ] ) = lo
        func([lo, lo,hi,hi ] ) = lo
        func([lo, lo,lo,hi ] ) = lo
        ...
        "reproduce the smallest variances from the input distributions"
        (probabilistically, we assume full correlation)

     """

    def _recursive_generator(cs, inv_func, factor=1, indices=[], i_last=-1):
        """
        generator to permute all input distributions and compute inverse-of-sum terms
        """
        #recurse into sets with more indices:
        factor_next = -1 * (factor+1*_np.sign(factor)) #1, -2, 3, -4, 5,....
        for i in range(i_last+1,len(cs)):
            indices.append(i)
            s=0.
            for idx in indices:
                s+=cs[idx]**2
            print(indices, s, factor)
            yield factor * inv_func(s)
            indices.pop()
        for i in range(i_last+1,len(cs)):
            indices.append(i)
            yield from _recursive_generator(cs, inv_func, factor_next, indices, indices[-1])
            indices.pop()
        #compute the current index set's p value:

    return inv_func(_np.sum(_recursive_generator(covariances, inv_func)))**0.5




