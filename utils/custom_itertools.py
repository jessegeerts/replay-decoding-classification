# -*- coding: utf-8 -*-
"""
Module contents:

    # groupby_
    # batched
        
@author: daniel
"""

from itertools import groupby
        
import numpy as np
import warnings
import sys


def combinations_2(a,b):
    """ combinations_2('ABC','XYZ') = ['AX','AY','AZ','BX','BY',...,'CZ']
    """
    for aa in a:
        for bb in b:
            yield aa,bb
            
def groupby_(a,*args):
    """
    for keys ``a`` and ``*args``, 0 or more iterables of matching length,
    yields a_val, args_1_iter, args_2_iter, ... args_n_iter
    
    unliek with groupby, the values need not be pre-sorted, we do the sorting here.
    we also handle all the keying and dropping the key etc.
    """
    
    vals = sorted(zip(a,*args),key=lambda x: x[0])
    
    for a_val,group in groupby(vals,lambda x: x[0]):
        yield (a_val,) + tuple(zip(*group)[1:])
        
        

def batched_range(n,b,print_progress=False):
    """ 
    Example:
    
        for start,end,blen in batched_simple(3067,100):
            out[start:end] = something(blen)

    All batches appart from the last are of size `b`. If `b` is less than `1` it
    is set to `1`.    
    """   
    b , n = int(b), int(n)
    b = 1 if b < 1 else b
    for s in xrange(0,n,b):
        e = min(s+b,n)
        yield s, e, e-s
        if print_progress:
            print("done: [0:{:d}], remaining: [{:d}:{:d}]".format(e,e,n))
            sys.stdout.flush()


def batched(counts,*args,**kwargs):
    """
    A generator that takes a numpy array of counts and possibly some other
    numpy arryays (say ``a`` and ``b``) of the same length.  The idea is that you 
    cant fully vectorise you calculations on ``a`` and ``b``, rather you have
    to take a pair of elements from ``a`` and ``b`` to produce outputs of varying size, known 
    in advance to be of length as specified in the corresponding element of ``counts``.
    Ideally you would like to iterate over ``a`` and ``b`` in one go, to produce
    a single large output array which you can then operate on at the end, however if 
    the counts involved are too large you will find that the processing has to be done in 
    "batches" to avoid ever creating a humongous array..hence this generator.  

    You specify the maximum batch len using the ``blen=100`` argument.
    
    Example::
    
        a,b = arange(400),rand(400)
        counts = work_out_output_size(a,b)
        hist_data = zeros(some_shape)
        for blen, out_slices, a_batch, b_batch in batch(counts,a,b,blen=300):
            batch = empty(blen)
            for slc,a_val,b_val in zip(out_slices,a_batch,b_batch):
                batch[slc] = some_function(a_val,b_val)
            hist += bincounts(batch)
            
    To be clear about the above, the yeilded value ``blen`` is an integer giving
    the sum of counts for the given batch. ``a_batch`` is a single slice into the original
    ``a`` array, corresponding to the elements to be processed in this batch, and
    simillarly for ``b_batch``.  ``out_slices`` is an array of slices, which 
    essentially partiion an array of length ``blen`` into section of length as
    requested in the corresponding element of ``counts``.
            
    The kwarg flag ``as_slices`` is by defualt ``True``, which gives the behaviour
    described above.  If you set it to ``False`` the ``out_slice`` is replaced
    by counts_batch, which is exactly analagous to ``a_batch``.
    
    This is quite specific behabiour, but it's what I needed.
    """
    blen = kwargs.get('blen',100)
    as_slices = kwargs.get('as_slices',True)
    
    cum_counts = np.cumsum(counts)
    
    start = 0
    cum_off = 0
    while start < len(counts):
        # we want to get a slice counts[start:end+1], such that 
        # the sum is <=blen, and any longer slice would be >blen
        end = start + np.searchsorted(cum_counts[start:], cum_off+blen)
        if end == len(cum_counts):
            end -=1
        elif cum_counts[end]-cum_off > blen:
            if end == start:
               warnings.warn("oversized batch") 
            elif cum_counts[end] - cum_off > blen:
                end -= 1
        
        if as_slices:
            # make the partition/slices list
            n = end-start +1
            out_2 = np.empty(n,dtype=object)        
            sub_off = 0
            for ii in range(n):
                out_2[ii] = slice(sub_off,sub_off+counts[start+ii])
                sub_off += counts[start+ii]
        else:
            out_2 = counts[start:end+1]
            
        yield (cum_counts[end]-cum_off, out_2) + tuple([var[start:end+1] for var in args])
        
        start = end+1
        cum_off = cum_counts[end]