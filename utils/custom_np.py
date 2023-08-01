# -*- coding: utf-8 -*-
"""
Module contents:

    # labeledCumsum
    # next_power_of_2 

@author: daniel, modified by Jesse Geerts, March 2022
"""

import numpy as np
from numpy import isnan,cumsum,insert,nan
import math
from contextlib import contextmanager

@contextmanager
def ignore_divide_by_zero():
    """ Usage:
    
        with ignore_divide_by_zero():
            a = b/c
    """
    old = np.seterr(divide='ignore',invalid='ignore')    
    yield
    np.seterr(**old)
    

def labeledCumsum(X,L,invalid=nan):
    """    
    Based on Matalb code written by DM, this python verison also written by DM.
    
     e.g. X=[3   5 8  9 1  2    5    8 5  4  9   2]
          L=[0   1 1  2 2  0    0    1 1  1  0   5]
     result=[NaN 5 13 9 10 NaN  NaN  8 13 17 NaN 2]

     where `L==0`, the output is set to `invalid`     
     Conversion from Matlab was done without massive amounts of testing, but it seems correct.
     
     `X` can also be a scalar such as `1`, in which case we treat it as an array
     of that value repeated `len(L)` times.
    """
    
    L = L.ravel()
    
    if not np.isscalar(X):
        X = X.ravel()
        
        if len(L) != len(X):
            raise Exception('The two inputs should be vectors of the same length.')
    
        # Do the full cumulative sum
        X[isnan(X)] = 0
        S = cumsum(X)
    else:
        # this is a slight optimisation/convenience
        S = np.arange(1,len(L)+1,dtype=np.float32)
        if X != 1:
            S *= X
    
    mask = L.astype(bool)
    
    # Lookup the cumulative value just before the start of each segment
    isStart = mask.copy()
    isStart[1:] &= (L[:-1] != L[1:])
    startInds, = isStart.nonzero()
    S_starts = S[startInds-1] if startInds[0] != 0 else  insert(S[startInds[1:]-1],0,0)
    
    # Subtract off the excess values (i.e. the ones obtained above)
    L_safe = cumsum(isStart) # we do this in case the labels in L were not sorted integers
    S[mask] = S[mask] - S_starts[L_safe[mask]-1]  
    
    # Put NaNs in the false blocks
    S[L==0] = invalid
    
    return S

def labelMask1D(M):
    """ e.g.
    M =      [F T T F F T F F F T T T]
    result = [0 1 1 0 0 2 0 0 0 3 3 3]
    
    M is boolean array, result is integer labels of contiguous True sections."""
    
    if M.ndim != 1:
        raise Exception("this is for 1d masks only.")
        
    is_start = np.empty(len(M),dtype=bool)
    is_start[0] = M[0]
    is_start[1:] = ~M[:-1] & M[1:]
    L = cumsum(is_start)
    L[~M] = 0
    return L


def is_power_of_2(x):
    """ from http://graphics.stanford.edu/~seander/bithacks.html """
    x = np.asanyarray(x,dtype=int)
    return np.logical_and(x , np.logical_not(x & (x - 1)) )

def next_power_of_2(n):
    # this is not suppsoed to be super fast or anything
    return 2 ** math.ceil(math.log(n, 2))
    

def reindex_masked(M,*args):
    """For a 1D mask `M`, and one or more index vectors of the same length, `X1`,`X2`,...
    we filter and re-label the indices for the following usage:
    
       Y = # some array of length n
       X1 = # some array of indices into Y
       M = # some bool mask of lenght n
       Y_new = Y[M]
       X1_new = reindex_masked(M,X1)
       all(Y_new[X1_new] == Y[X1][M[X1]])
    
    Note that in many cases you can use the `Y[X1][M[X1]]` method rather than
    explicitly relableing `X1_new`, however sometimes you will prefer to call
    this `reindex_masked` function, for example when you want to pass `Y_new` and
    `X1_new` to a function which should be agnostic of any masking.
    
    If you have and multiple `Xi` the syntax is: `reindex_masked(M,X1,X2,X3...)` 
    """
    M = M.astype(bool)
    new_inds = np.cumsum(M)
    return tuple((new_inds[X[M[X]]-1] for X in args))
    
#@profile
def repeat_ind(n):
    """By example:
    
        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [2, 2, 2, 5, 5, 7, 7, 8]
        
    That is the input specifies how many times to repeat the given index.

    It is equivalent to something like this :
    
        hstack((zeros(n_i,dtype=int)+i for i, n_i in enumerate(n)))
        
    But this version seems to be faster, and probably scales better, at
    any rate it encapsulates a task in a functoin.
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")

    """        
    # Old implementation, without using np.repeat
    
    n_mask = n.astype(bool)
    n_inds = np.nonzero(n_mask)[0]
    n_inds[1:] = n_inds[1:]-n_inds[:-1] # take diff and leave 0th value in place
    n_cumsum = np.empty(len(n)+1,dtype=int)
    n_cumsum[0] = 0 
    n_cumsum[1:] = np.cumsum(n)
    ret = np.zeros(n_cumsum[-1],dtype=int)
    ret[n_cumsum[n_mask]] = n_inds # note that n_mask is 1 element shorter than n_cumsum
    return cumsum(ret)
    """
    
    return np.repeat(np.arange(len(n)), n)


def count_to(n):
    """By example:
    
        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]

    That is it is equivalent to something like this :
    
        hstack((arange(n_i) for n_i in n))
        
    This version seems quite a bit faster, at least for some
    possible inputs, and at any rate it encapsulates a task 
    in a function.
    """
    if n.ndim != 1:
        raise Exception("n is supposed to be 1d array.")
        
    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1]+1,dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask] 
    ret[0] -= 1
    return np.cumsum(ret)[:-1]
    
def duplicates_as_complex(x,already_sorted=False):
    """
    By example:
        x = [9.9    9.9     12.3    15.2    15.2    15.2    ]
        ret=[9.9+0j 9.9+1j  12.3+0j 15.2+0j 15.2+1j 15.2+2j ]
        
    That is we return `x` as a sorted complex number, where 
    the real part is simply the values in `x`, but the 
    imaginary part is a sub-index within a block of duplicates,
    i.e. if value `v` occurs three times, it appears as
    `v+0j, v+1j, v+2j`.
    
    The advantage of this system is that it can be used in a 
    natural way with `np.sort` and `np.searchsorted`.
    """
    if not already_sorted:
        x = np.sort(x)
    is_start = np.empty(len(x),dtype=bool)
    is_start[0], is_start[1:] = True, x[:-1] != x[1:]
    labels = np.cumsum(is_start)-1
    sub_idx = np.arange(len(x)) - is_start.nonzero()[0][labels]
    return x + 1j*sub_idx


def argsort_inverse(x,**kwargs):
    """ Replaces each value in x with it's ranked index.
    Note this probably only works with 1d data as it currently stands.
    """
    ret = np.empty(x.shape,dtype=x.dtype)
    ret[np.argsort(x,**kwargs)] = np.arange(len(x))
    return ret


def shift(arr, num, fill_value):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result