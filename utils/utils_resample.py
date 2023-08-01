# -*- coding: utf-8 -*-
import numpy as np
from numpy_groupies import aggregate_np as aggregate
import numpy_groupies as npg

def resample_labels(labels_a, labels_b, values_a, values_b, bin_size):
    """
    Does a very specifc kind of resampling/relabeling.
    
    labels_a and values_a are matching 1D arrays, as are labels_b and values_b.

    Here we aggregate (using mean) over each labeled group of values, and then
    bin up the results using bin_size. Then we iterate over the bins, and zero
    out some of the labels in either a or b so as to get the same number of
    groups for the given bin.
    
    For example a and b might correspond to a pair of trials, and values_a and 
    values_b give the speed in those trials; labels_a would have one value
    for eahc pos sample and might identify runs through grid fields:
        [0 0 0 1 1 1 1 0 0 0 2 2 .... 95 95 95 0 0 0]
    labels_b would do something similar for trial b.
    
    The result would then be similar to the input labels_a and labels_b, but
    with more zeros, so that mean run speed has the same distribution for trial
    a and trial b.  Note that labels are reasigned to be 1:n (with, as a side
    effect of the strict resampling rules, n now being equal for both a and b).
    """

    group_agg_a = aggregate(labels_a, values_a, func='mean')
    group_agg_b = aggregate(labels_b, values_b, func='mean')
    group_bin_a = (group_agg_a/bin_size).astype(int)
    group_bin_b = (group_agg_b/bin_size).astype(int)
    keep_group_a = np.ones(len(group_bin_a), dtype=bool)
    keep_group_b = np.ones(len(group_bin_b), dtype=bool)
    max_bins = max(np.max(group_bin_a), np.max(group_bin_b))
    groups_in_bin_a = aggregate(group_bin_a[1:], np.arange(1, len(group_bin_a)),
                                               func='array', fill_value=[], size=max_bins+1)
    groups_in_bin_b = aggregate(group_bin_b[1:], np.arange(1,len(group_bin_b)),
                                               func='array', fill_value=[], size=max_bins+1)
    for group_a, group_b in zip(groups_in_bin_a, groups_in_bin_b):
        if len(group_a) > len(group_b):
            np.random.shuffle(group_a) # TODO: isn't there a 1-liner way of resampling?
            keep_group_a[group_a[len(group_b):]] = False
        elif len(group_b) > len(group_a):
            np.random.shuffle(group_b)
            keep_group_b[group_b[len(group_a):]] = False
    labels_a = npg.relabel_groups_masked(labels_a, keep_group_a)            
    labels_b = npg.relabel_groups_masked(labels_b, keep_group_b) 
    return labels_a, labels_b