# -*- coding: utf-8 -*-
"""
This is all rather perliminary. Do not take it too seriously.
"""

import numpy as np
import numpy_groupies as npg
import matplotlib.pylab as plt
import scipy.spatial.distance as sp_dist
from scipy.signal import argrelmax as sp_argrelmax
from utils.custom_mpl import axes_square
from collections import namedtuple
from utils.custom_functools import append_docstring

"""
    all pos_* stuff is n_pos long
    all runA_* are same length, with indices refering to run labels version A
    all runB_* are same length, with indices refering to run labels version B
    all runC_* are same length, with indices refering to run labels version C
    
    Note that C doesn't include the zero label whereas A and B do.
    
    this A/B/C thing is needed because we filter runs during the computation and
    update the run labeling.
    Note that *_end indexing is inclusive, so you must +1 when using in slices.
"""

InfoPeakRunRateOut = namedtuple('InfoPeakRunRateOut', ("pos_rate pos_field_label "
                                                       "pos_run_label_A pos_run_label_B pos_run_thresh_B pos_peak_label_B "
                                                       "runB_peak_idx runB_peak_rate runB_is_good runC_peak_idx "
                                                       "runC_peak_rate runC_peak_width runC_peak_start runC_peak_end "
                                                       "runB_run_width min_rate_peak_hz"))
InfoPosPeak = namedtuple('InfoPosPeak', ("pos_rate peakA_idx "
                                         "peakA_left_idx peakA_right_idx peakA_is_distinct peakB_peak_width_spa "
                                         "min_rate_peak_hz"))


def get_pos_peak_info(self, t, c, min_rate_peak_hz=2.0, peak_frac_thresh=0.5,
                      max_width_seconds=10,  # this is actually the max half-width
                      returnMore=False, rate_kwargs={}):
    """
    ``min_rate_peak_hz`` can be the string ``'mean'`` or ``median`` or an actual
    value, ``mean`` and ``median`` are applied to ``pos_rate``.
    """
    # get the temporally smoothed rate
    pos_rate = self.get_t_rate(t, c, **rate_kwargs)

    if min_rate_peak_hz == "mean":
        min_rate_peak_hz = np.mean(pos_rate)
    elif min_rate_peak_hz == "median":
        min_rate_peak_hz = np.median(pos_rate)

    # find peaks that are greater than min_rate_peak_hz, and sort into descending order
    pos_rate_copy = pos_rate.copy()
    pos_rate_copy[pos_rate < min_rate_peak_hz] = 0
    peaks_idx, = sp_argrelmax(pos_rate_copy)
    peaks_rate = pos_rate[peaks_idx]
    sort_idx = np.argsort(peaks_rate)[::-1]  # descending
    peaks_idx = peaks_idx[sort_idx]
    peaks_rate = peaks_rate[sort_idx]

    # starting with the largest peaks, find the left and right edge of the peak
    # which are the points where the rate drops down below half the peak height
    # if this region overlaps with a precious peak then consider this peak non-"distinct"
    # (but the previous peak is).  Future (i.e. smaller) peaks overlapping with non-distinct peaks
    # are also non-distinct.
    pos_is_peak = np.zeros(int(self.n_pos), dtype=bool)
    max_width = int(self.pos_samp_rate * max_width_seconds)
    peak_left_idx = np.empty(len(peaks_idx), dtype=int)
    peak_right_idx = np.empty(len(peaks_idx), dtype=int)
    peak_is_distinct = np.empty(len(peaks_idx), dtype=bool)
    for ii, (p_idx, p_rate) in enumerate(zip(peaks_idx, peaks_rate)):
        # find left edge
        is_below_thresh = pos_rate[p_idx - max_width:p_idx] < p_rate * peak_frac_thresh
        peak_left_idx[ii] = left_idx = \
            max(0, p_idx - max_width) + (is_below_thresh.nonzero()[0][-1]
                                         if np.any(is_below_thresh) else 0)
        # find right edge        
        is_below_thresh = pos_rate[p_idx:p_idx + max_width] < p_rate * peak_frac_thresh
        peak_right_idx[ii] = right_idx = \
            p_idx + (is_below_thresh.nonzero()[0][0]
                     if np.any(is_below_thresh) else max_width)

        # check if this peak is distinct, and record its footprint for future peaks        
        peak_is_distinct[ii] = ~np.any(pos_is_peak[left_idx:right_idx])
        pos_is_peak[left_idx:right_idx] = True

    peakB_left_idx = peak_left_idx[peak_is_distinct]
    peakB_right_idx = peak_right_idx[peak_is_distinct]

    xy = self.xy
    peakB_peak_width_spa = np.full(len(peakB_left_idx), np.nan)
    for ii, (ii_start, ii_end) in enumerate(zip(peakB_left_idx, peakB_right_idx)):
        dists = sp_dist.pdist(xy[:, ii_start:ii_end + 1].T, 'sqeuclidean')
        peakB_peak_width_spa[ii] = np.sqrt(np.max(dists))

    if returnMore:
        return InfoPosPeak(pos_rate=pos_rate,
                           peakA_idx=peaks_idx,
                           peakA_left_idx=peak_left_idx,
                           peakA_right_idx=peak_right_idx,
                           peakA_is_distinct=peak_is_distinct,
                           peakB_peak_width_spa=peakB_peak_width_spa,
                           min_rate_peak_hz=min_rate_peak_hz)
    else:
        return peakB_peak_width_spa, peaks_rate[peak_is_distinct]


@append_docstring(get_pos_peak_info)
def plot_pos_peak_info(self, t=None, c=None, info=None, **kwargs):
    if info is None:
        info = get_pos_peak_info(self, t=t, c=c, returnMore=True, **kwargs)

    plt.cla()

    plt.plot([0, self.n_pos], [info.min_rate_peak_hz] * 2, 'y', lw=2)
    times = np.arange(self.n_pos) / float(self.pos_samp_rate)
    plt.plot(times, info.pos_rate, 'r')
    peakB_idx = info.peakA_idx[info.peakA_is_distinct]
    peakB_start = info.peakA_left_idx[info.peakA_is_distinct]
    peakB_end = info.peakA_right_idx[info.peakA_is_distinct]

    plt.plot(times[peakB_idx], info.pos_rate[peakB_idx], 'ko')

    plt.plot(times[peakB_start], info.pos_rate[peakB_start], 'c>')
    plt.plot(times[peakB_end], info.pos_rate[peakB_end], 'c<')
    plt.xlim(0, self.duration)
    plt.xlabel('seconds')


def get_run_peak_info(self, t, c, min_rate_peak_hz=2.0, peak_frac_thresh=0.5,
                      mode_fields=3, wing_run_frac=0.2, returnMore=False):
    """
    ``mode_fields`` is passed through to _getFieldLabelAtXY.
    ``min_rate_peak_hz`` is used to filter runs that don't have a high enough
    rate, this takes us from label A to label B. ``peak_frac_thresh`` defines
    the width of the peak relative to its peak height.
    ``wing_run_frac`` defines how much of the peak is alowed to bleed over the
    start/end boudnary of the run, specified as a fraction of the run length.  Each
    edge is allowed to be this large, ie. total peak outside run can be 2x this value.

    Some of the ouputs are only provided when ``returnMore`` is True..    
    ``runB_run_width``.
    
    """
    if wing_run_frac > 1:
        raise NotImplementedError("the slicing is currently implemented stupidly, with 1x run width as search space")
    bad_pos_idx = self.n_pos + 100

    # get the temporally smoothed rate
    pos_rate = self.get_smoothed_rate(t, c)

    # Get the initial run labels, which we will subsequently be filtering
    pos_field_label = self.get_field_label_at_xy(t, c, mode='pos', mode_fields=mode_fields)
    pos_run_label_A = npg.label_contiguous_1d(pos_field_label)

    # Filter runs, removing runs with small peak rates. Relabel runs to labels B.    
    runA_peak_rate = npg.aggregate_np(pos_run_label_A, pos_rate, func='max',
                                      fill_value=np.nan)
    runA_is_good = runA_peak_rate > min_rate_peak_hz
    pos_run_label_B = npg.relabel_groups_masked(pos_run_label_A, runA_is_good)

    # Get the peak value and index for the peaks in run labels B
    runB_peak_rate = npg.aggregate_np(pos_run_label_B, pos_rate, func='max',
                                      fill_value=np.nan)
    runB_peak_idx = npg.aggregate_np(pos_run_label_B, pos_rate, func='argmax',
                                     fill_value=bad_pos_idx)
    runB_peak_rate[0] = np.nan
    runB_peak_idx[0] = bad_pos_idx
    n_run_B = len(runB_peak_idx)

    # Label the area around the peak in each run which is >= peak_height*frac,
    # use the same labels B as the runs themselves.
    pos_run_thresh_B = runB_peak_rate[pos_run_label_B] * peak_frac_thresh
    pos_is_in_peak = pos_rate >= pos_run_thresh_B
    pos_peak_label_B = pos_run_label_B.copy()
    pos_peak_label_B[~pos_is_in_peak] = 0
    pos_peak_label_B = npg.label_contiguous_1d(pos_peak_label_B)
    # at this point, run_peak_label has 1 or more labeled blocks per run, using different labels to the run labels
    # we want to drop all blocks except those that contain an actual peak,
    # and at the same time restore the original run labeling
    relabel = np.zeros(np.max(pos_peak_label_B) + 1, dtype=int)
    relabel[pos_peak_label_B[runB_peak_idx[1:]]] = np.arange(1, n_run_B + 1)
    pos_peak_label_B = relabel[pos_peak_label_B]

    # Now we need to find the width of the "wings" of the peaks, where the peaks extend 
    # out beyond the edges of the runs.
    runB_start_idx = npg.aggregate_np(pos_run_label_B, np.arange(self.n_pos), func='first')
    runB_end_idx = npg.aggregate_np(pos_run_label_B, np.arange(self.n_pos), func='last')
    runB_length = runB_end_idx - runB_start_idx + 1

    runB_has_left_wing = pos_peak_label_B[runB_start_idx].astype(bool)
    runB_has_left_wing[0] = False
    runB_has_right_wing = pos_peak_label_B[runB_end_idx].astype(bool)
    runB_has_right_wing[0] = False
    runB_left_wing_width = np.full(n_run_B, np.nan)
    runB_right_wing_width = np.full(n_run_B, np.nan)

    # you  could possibly vectorise this using np.repeat but for now lets do this pair of ugly loops
    for ii_label, ii_start, ii_length, ii_thresh in zip(runB_has_left_wing.nonzero()[0],
                                                        runB_start_idx[runB_has_left_wing],
                                                        runB_length[runB_has_left_wing],
                                                        runB_peak_rate[runB_has_left_wing] * peak_frac_thresh):
        is_sub_thresh = pos_rate[max(0, ii_start - ii_length):ii_start][::-1] < ii_thresh
        if np.any(is_sub_thresh):
            runB_left_wing_width[ii_label] = is_sub_thresh.nonzero()[0][0]

    for ii_label, ii_end, ii_length, ii_thresh in zip(runB_has_right_wing.nonzero()[0],
                                                      runB_end_idx[runB_has_right_wing],
                                                      runB_length[runB_has_right_wing],
                                                      runB_peak_rate[runB_has_right_wing] * peak_frac_thresh):
        is_sub_thresh = pos_rate[ii_end:ii_end + ii_length] < ii_thresh
        if np.any(is_sub_thresh):
            runB_right_wing_width[ii_label] = is_sub_thresh.nonzero()[0][0]

    # Now we can do the filtering based on the presence/width of the wings
    runB_is_good = ((~runB_has_left_wing) | (runB_left_wing_width < runB_length * wing_run_frac)) & \
                   ((~runB_has_right_wing) | (runB_right_wing_width < runB_length * wing_run_frac))
    runB_is_good[0] = False

    runC_peak_start = npg.aggregate_np(pos_peak_label_B, np.arange(self.n_pos), func='first')
    runC_peak_end = npg.aggregate_np(pos_peak_label_B, np.arange(self.n_pos), func='last')
    mask = runB_left_wing_width < runB_length * wing_run_frac
    runC_peak_start[mask] -= runB_left_wing_width[mask]
    mask = runB_right_wing_width < runB_length * wing_run_frac
    runC_peak_end[mask] += runB_right_wing_width[mask]
    runC_peak_start = runC_peak_start[runB_is_good]
    runC_peak_end = runC_peak_end[runB_is_good]
    n_run_C = len(runC_peak_end)  # IMPORTANT: note that we drop group zero entierly for C stuff

    # Now we can get the peak properties for the runs, version C
    # here we define peak width as the maxium euclidean ditance between
    # pos samples within the region defined as the peak.
    runC_peak_rate = runB_peak_rate[runB_is_good]
    xy = self.xy
    runC_peak_width = np.full(n_run_C, np.nan)
    for ii, (ii_start, ii_end) in enumerate(zip(runC_peak_start, runC_peak_end)):
        dists = sp_dist.pdist(xy[:, ii_start:ii_end + 1].T, 'sqeuclidean')
        runC_peak_width[ii] = np.sqrt(np.max(dists))

    if returnMore:
        runB_run_start = npg.aggregate_np(pos_run_label_B, np.arange(self.n_pos), func='first')
        runB_run_end = npg.aggregate_np(pos_run_label_B, np.arange(self.n_pos), func='last')
        runB_run_width = np.full(n_run_B, np.nan)
        for ii, (ii_start, ii_end) in enumerate(zip(runB_run_start[1:], runB_run_end[1:])):
            if ii_start == ii_end:
                continue
            dists = sp_dist.pdist(xy[:, ii_start:ii_end + 1].T, 'sqeuclidean')
            runB_run_width[ii + 1] = np.sqrt(np.max(dists))
    else:
        runB_run_width = None

    return InfoPeakRunRateOut(pos_rate=pos_rate,
                              pos_field_label=pos_field_label,
                              pos_run_label_A=pos_run_label_A,
                              pos_run_label_B=pos_run_label_B,
                              pos_peak_label_B=pos_peak_label_B,
                              pos_run_thresh_B=pos_run_thresh_B,
                              runB_peak_idx=runB_peak_idx,
                              runB_peak_rate=runB_peak_rate,
                              runB_is_good=runB_is_good,
                              runC_peak_idx=runB_peak_idx[runB_is_good],
                              runC_peak_rate=runC_peak_rate,
                              runC_peak_width=runC_peak_width,
                              runC_peak_start=runC_peak_start,
                              runC_peak_end=runC_peak_end,
                              runB_run_width=runB_run_width,
                              min_rate_peak_hz=min_rate_peak_hz)


@append_docstring(get_run_peak_info)
def plot_run_peak_info(self, t=None, c=None, info=None, **kwargs):
    """
    provide t,c, and kwargs to pipe through to get_run_peak_info, or
    provide an info object to plot existing results.
    """
    if info is None:
        info = get_run_peak_info(self, t, c, **kwargs)

    plt.clf()
    plt.subplot(2, 1, 1)
    self.plot_pos_alt(colors=info.pos_field_label)
    plt.title('pos_field_label')
    axes_square()

    plt.subplot(2, 1, 2)
    plt.plot([0, self.n_pos], [info.min_rate_peak_hz] * 2, 'y', lw=2)
    plt.plot(info.pos_rate, 'r')
    plt.plot(info.runC_peak_idx, info.pos_rate[info.runC_peak_idx], 'ko')

    pos_run_thresh_C = info.pos_run_thresh_B.copy()
    pos_run_thresh_C[~info.runB_is_good[info.pos_run_label_B]] = False

    runC_peak_label = info.pos_peak_label_B
    runC_peak_label[~info.runB_is_good[info.pos_run_label_B]] = 0

    plt.plot(pos_run_thresh_C, 'k:')
    plt.plot((runC_peak_label > 0) * pos_run_thresh_C, 'm')

    plt.plot(info.runC_peak_start, info.pos_rate[info.runC_peak_start], 'c>')
    plt.plot(info.runC_peak_end, info.pos_rate[info.runC_peak_end], 'c<')

    plt.fill_between(np.arange(self.n_pos), info.runB_is_good[info.pos_run_label_B] * 0.45 - 1, -1, lw=0,
                     color='#ccccdd')
    plt.fill_between(np.arange(self.n_pos), (info.pos_run_label_B > 0) * 0.45 - 1.5, -1.5, lw=0, color='#ccccee')
    plt.fill_between(np.arange(self.n_pos), (info.pos_run_label_A > 0) * 0.45 - 2.0, -2, lw=0, color='#ccccff')
