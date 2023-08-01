# -*- coding: utf-8 -*-
"""
Module contents:
    # pearsonr_permutations
    # pearsonr_sign_bootstrap
    # permute_multi <-- used by pearsonr_permutations
DM, Feb 2015.
"""
 
import numpy as np
from custom_itertools import batched_range
from scipy.stats.mstats import gmean as geomean
import scipy.stats as stats
from scipy.stats import chisqprob

def mcnemar(x):
    """
    Computes the McNemar test.
    https://en.wikipedia.org/wiki/McNemar%27s_test
    c is a 1d-array, with indices corresponding to the following:

              test 2
              +     -        
           +  3  |  1
    test 1    -------
           -  2  |  0

    That is, you perform a boolean "test" twice on a each "subject", and
    record how many of the subjects get (+,+), how many get (+,-) and (-,+), (-,-).
    
    The mcnemar p-value tells you the probability that the two tests have 
    differing chances of success (i.e. across all "subjects").
    
    Note that only indices 3 and 0 are actually used in the test
    """
    d,b,c,a = x
    n = c+b
    if n < 25:
        p = stats.binom.cdf(b,n,0.5)
        return 2*p if p <0.5 else 2*(1-p)
    else:
        t = ((c-b)**2)/float(n)
        return chisqprob(t,1)
        
def pearsonr_permutations(x,y,k=100,tails=2):
    """ 
    For matching 1D vectors `x` and `y`, this returns `(r,p)` exactly like 
    `sp.stats.pearsonr`, except that here `p` is obtained using the 
    "permutation test", see:
    http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    
    This means we permute `y` with respect to `x`, `k` times, compute
    the correlation each time, and then see what fraction of these permuted `r`
    values exceed the original `r` value.  When `tails=2` we use `abs(r)` to
    perform this comparison, when `tails=1`, we check whether `r` is positive
    or negative and count only the number of larger or smaller permuted `r`s 
    respectively.

    See `permute_multi` in this module for notes on cached random values.
    It is this caching that makes this function so fast (once it has warmed up).
    """
    if x.ndim != 1 or x.shape != y.shape:
        raise Exception("Expected x and y to be a pair of matching 1d arrays.")
    k = int(k)
    n = len(x)        
    
    # we must request sufficient precision for what lies ahead
    x,y = x.astype(np.float64), y.astype(np.float64) 
    
    # compute the stuff that is invariant of y permutations
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.dot(x,x)
    Syy = np.dot(y,y)        
    denom = np.sqrt(n*Sxx-Sx**2) * np.sqrt(n*Syy-Sy**2) 
    
    # compute the unshufled Sxy
    Sxy_original = np.dot(x,y)    

    # for k permutations of y, compute Sxy
    Sxy_shuffles = np.empty(k,dtype=Sxy_original.dtype)
    for i_start,i_end,k_ in batched_range(k,n/60000.0,print_progress=True):
        inds = np.empty((k_,n)) # TODO: may want to move this out the loop
        for i in xrange(k_):
            inds[i,:] = np.random.permutation(n)
        Sxy_shuffles[i_start:i_end] = np.dot(x, permute_multi(y,k_).T)
        
    # Compute the r value for the original and the shuffles 
    r_original = (n*Sxy_original - Sx*Sy)/denom 
    r_shuffles = (n*Sxy_shuffles - Sx*Sy)/denom 
    
    if tails == 2:
        p = np.sum(abs(r_shuffles) > abs(r_original))
    elif tails == 1 and r_original > 0:
        p = np.sum(r_shuffles > r_original)
    elif tails == 1 and r_original < 0:
        p = np.sum(r_shuffles < r_original)
    else:
        raise Exception("tails should be 1 or 2, and r must be non-zero")
    
    return r_original, float(p)/float(k)

def pearsonr_sign_bootstrap(x,y,k=1000):
    """ 
    For matching 1D vectors `x` and `y`, this returns `(s,p)` much like 
    `sp.stats.pearsonr`, except that here `s` is just the sign of `r`, rather than
    its exact value, and `p` is obtained as follows:
    
    Pick `n` samples from the list of `n` `(x,y)` pairs (i.e. with replacement)
    and compute `s`.  Repeat this `k` times, and define `p` as the fraction of 
    `s` values which have the opposite sign to the original (un-resampled) `s`.
    
    Note that we don't need to compute the two standard deviations in the denominator
    because they are both positive (well, if we ignore the possibility of equaling zero)
    and thus do not effect the sign of the pearson as a whole.
    
    TODO: perhaps we should deal properly with the case where the resampled x or
    y has 0 standard deviation. In such cases we ought to throw a divide-by-zero
    error, or something.
        
    DM, Feb 2015
    """
    if x.ndim != 1 or x.shape != y.shape:
        raise Exception("Expected x and y to be a pair of matching 1d arrays.")
        
    k = int(k)
    n = len(x)        

    # prepare a matrix with 3 columns: [x  y  x*y]
    x_y_xy = np.hstack(( \
            x[:,np.newaxis],
            y[:,np.newaxis],
            (x*y)[:,np.newaxis]))
    
    # compute the original [Sx Sy Sxy]
    S_x_y_xy_original = np.sum(x_y_xy, axis=0)
 
    # for k resamplings, compute [Sx Sy Sxy]
    S_x_y_xy_shuffles = np.empty(shape=(k,3))
    for i in xrange(k):
        S_x_y_xy_shuffles[i,:] = np.dot(np.bincount(np.random.randint(n,size=n),minlength=n),
                                        x_y_xy)
    
    # Compute the s value for the original and the shuffles 
    s_original = np.sign(n*S_x_y_xy_original[2] - S_x_y_xy_original[0] * S_x_y_xy_original[1])
    s_shuffles = np.sign(n*S_x_y_xy_shuffles[:,2] - S_x_y_xy_shuffles[:,0] * S_x_y_xy_shuffles[:,1])
    
    # work out what fraction of the shuffles have the opposite sign to the original
    p = np.sum(s_shuffles != s_original)
    
    return int(s_original), float(p)/float(k)
    
    
    
def randint_(n, shape, _cache={}, cache_cmd=None):
    k = np.product(shape)
    
    if _cache.get('n',0) < n:
        _cache['inds'] = cached_inds =  np.random.randint(n,size=k)
        _cache['n'] = n
    else:
        cached_inds = _cache['inds']
        

    if _cache.get('n',0) == n:
        inds = cached_inds
    elif len(cached_inds) > k:
        inds = cached_inds.compress(cached_inds < n)
    else:
        inds = []
        
    if len(inds) < k:
        raise NotImplementedError("this can easily happen but it's difficult to " +\
            "know what the best caching behaviour is here...ideally should store " +\
            "the full 0-RAND_MAX numbers and then convert to 0-n as needed. See" + \
            "https://github.com/numpy/numpy/blob/4cbced22a8306d7214a4e18d12e651c034143922/numpy/random/mtrand/randomkit.c#L260")
    else:
        inds = inds[:k]
        
    return inds.reshape(shape)
        
def permute_multi(X, k, _cache={}, cache_cmd=None):
    """For 1D input `X` of len `n`, it generates an `(k,n)` array
    giving `k` permutations of `X`.

    When used repeatedly with similar n values, this function will be
    fast as it caches the permutation indicies. In (almost?) all cases
    this will be fine, however we take the precutionary measure of
    pre-permuting the data before applying the cached permutations,
    where the extra permutation is unique each time.  Note that strictly
    speaking, this may still not be truly sufficient: if the correlation 
    between rows in the cached indices happens to be quite a long way 
    from the expectation, then you are effecitvely using less than `k`
    permutations, wich would be fine if you did it once, but if you use
    that many times, then any meta-analysis will possibly have signficantly
    less power than it appears to...note that without caching this could also 
    happen but the probabilities grow combinatorially, so very quickly become
    infitesimal...thus this discussion really is only relveant for small `k`
    and `n`. For small `k`, you could partly gaurd agaisnt this by artificially
    using a larger `k` and then randomly subsampling rows.
    
    If you set `cache_cmd=None`, the cache will be used. 
    For `cache_cmd=-1` the cache will not be used.
    For `cache_cmd=+1` the cache will be reset and then used.
    TODO: we possibly want to apply some heuristic to shrink the cached
    array if it appears to be too big in the vast majority of cases.
    
    Ignoring the caching, this function is just doing the following:
    
        def permute_multi(X,k):
            n = len(X)
            ret = np.empty(shape=(k,n),dtype=X.dtype)
            for i in xrange(k):
                ret[i,:] = np.random.permutation(X)
            return ret
            
    
    """
    
    if cache_cmd == -1:
        _cache = {} # local dict will go out of scope at end of function
    elif cache_cmd == +1:
        _cache['inds'] = np.array([[]]) # reset the cached inds
        
    cached_inds = _cache.get('inds',np.array([[]]))

    n = len(X)
    
    # make sure that cached_inds has shape >= (k,n)
    if cached_inds.shape[1] < n:
        _cache['inds'] = cached_inds = np.empty(shape=(k,n),dtype=int)
        for i in xrange(k):
            cached_inds[i,:] = np.random.permutation(n)
    elif cached_inds.shape[0] < k:
        raise NotImplementedError("TODO: need to generate more rows")
    
    inds = cached_inds[:k,:] # dispose of excess rows
    
    if n < cached_inds.shape[1]:
        # dispose of high indices
        inds = inds.compress(inds.ravel()<n).reshape((k,n))

    # To prevent spurious patterns in the inds mapping back to the data in a
    # consistent way, we compose all the cached permutations with a novel, unique,
    # permutation, i.e. we perform the novel permutation first and then apply
    # the other permutations to that.
    X = np.random.permutation(X)    
    return X[inds] 
    
#@profile
def pearsonr_binned(x,y,k=100,percent=5,max_x=None,min_x=0):
    """    
    Returns:    
        (`p`, `x_bin_centers`, `y_means`)
     
    Any `x` greater than `max_x` or smaler than `min_x` are discarded. 
    If `max_x` is `None` then `max_x` is set to `max(x)`. 

    We then split the x values into m bins of even width, with m as large as 
    possible such that each bin still contains at least `percent` of the data.
    
    We then sample `c` values (with replecament) from each bin, and take
    the mean, this is the bin's y-value. The x-value is the centre of the bin. 
    The value of `c` is chosen to be the number of points in the least-occupied bin.
    
    We compute the sign of the pearson correlation for those means.  Then we
    repeat the resamplign and sign-of-pearson calculation `k` times. The fraction
    of these sign values which are positive is returned as `p`.    
    """
    if x.ndim != 1 or x.shape != y.shape:
        raise Exception("exepcted 1d x , with matching y")
        
    max_x = max_x if max_x is not None else np.max(x)
    good = (x <= max_x) & (x >= min_x) 
    x, y  = x[good], y[good]

    sort_inds = np.argsort(x)
    x, y = x[sort_inds], y[sort_inds]
    
    n = len(x)
    m_bins = 2
    thresh = int(np.ceil(n*float(percent)/100.0)) # must be >= to this
    for m_bins in xrange(2,n):
        partition_inds = np.searchsorted(x,np.linspace(0,max_x,m_bins+1)) # +1 is needed to get m_bins
        if np.min(np.diff(partition_inds)) < thresh:
            m_bins -= 1 # we've gone one step too far
            break

    partition_inds = np.searchsorted(x,np.linspace(0,max_x,m_bins+1))
    bin_counts = np.diff(partition_inds)
    c = np.min(bin_counts) # smallest count of all the bins

    # we use bin centers do give x values
    x_bin_centres = (np.arange(m_bins) + 0.5) * (float(max_x)/m_bins)
    Sx = np.sum(x_bin_centres)

    s = np.empty(k,dtype=bool)
    ymeans = np.empty((m_bins,k))
    for i_start,i_end,k_ in batched_range(k,n/60000.0,print_progress=True):
        # generate m_bins x ck array of inds into the original 1d vectors
        inds = randint_multi(bin_counts,k_*c)
        inds += partition_inds[:-1][:,np.newaxis]
    
        # use the inds to take values from y, and then compute means over c
        yvals = y[inds].reshape((m_bins,c,k_))
        ymeans_ = ymeans[:,i_start:i_end] = np.mean(yvals,axis=1) # this is now m_bins by k_
       
        Sy = np.sum(ymeans_,axis=0)
        Sxy = np.dot(x_bin_centres,ymeans_)
        
        s[i_start:i_end] = (m_bins*Sxy > Sx*Sy)
        
    
    return np.mean(s>0), x_bin_centres, ymeans
    

def randint_multi(n,k):
    """
    we generate `len(n)xk` array, where the values in the ith column
    are random integers,x, with 0 < x < `n[i]`, that is `n` is a 1d vector 
    specifying the maximum for each column.

    Note: I tried caching random numbers and vectorising stuff, but it was
    about 4 times slower.  This might not be so bad in numpy 1.10 once integer
    arithmetic is vectorised, but it seems the simplest thing is already pretty 
    good.
    """    
    return np.vstack((np.random.randint(nn,size=k) for nn in n))

    