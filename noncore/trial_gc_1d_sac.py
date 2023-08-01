# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
from utils.custom_functools import append_docstring
from numpy.polynomial.polynomial import polyfit
from scipy.signal import savgol_filter
from collections import namedtuple

from utils.custom_mpl import trans_xdata_yaxes, trans_xaxes_ydata
from utils.custom_exceptions import PointsOfInterestNotFound, DontTrustThisAnalysisException
  
from noncore.trial_gc import TrialGCAnalysis as AnalysisClass # used only for docstrings
                   
Info1dACProps = namedtuple("Info1dACProps", ("scale clusteriness peak_width "
                                         "trough ac1d dx bin_size_cm min_scale "
                                         "max_scale before after thresh peak_top "
                                         "peak_bottom top_sub bottom_sub coef "
                                         "half_height half_width"))

InfoDrift1d = namedtuple('InfoDrift1d', ("a b var_a best_off mean_diff score "
                                         "bin_size_cm"))

class TrialGC1dSACProps(object):
    """This is a mixin for trial, which depends on TrialGCAnalysis."""
    
    @append_docstring(AnalysisClass.get_spa_ac_windowed)
    def get_spa_ac_1d_props(self, t, c, bin_size_cm=1.0, max_scale=100.0,
                            min_scale=5.0, win_size_sec='trial', win_size_cm=None,
                            down_samp_rate=25, smoothing=31.0, sg_polyn=2,
                            thresh=0.01, peak_fudge_cm=5.0, ac1d=None, 
                            return_extra=False):
        """
        max and min scales are not neccessarily exact, we just need limits for simplicty in the calculation.
        see ``TT.get_spa_ac_windowed`` in ``trial_gc.py`` for the meaning of the ``win_size_sec`` and ``down_samp_rate`` args.
        
        ``smoothing`` is the width in cms of the savistkay golay filter applied to the ac1d to get the first derivative.
        it will be rounded up to the nearest odd number of bins
        ``sg_polyn``gives the polynomial order of the S-G filter.
        
        TODO: check whether S-G derivative is being evaluated half way between bins or at bin left...etc.
        
        returns: 
        (a) scale in cm;   
        (b) steepest greadient in central peak;   
        (c) width of peak around scale as a fraciton of scale.
        
        Note that steepest gradient is normalised to cell's np.mean rate and to scale 
        (first part of the normalisation is done on the 1d autocorr not on its gradient)
        
        Note ``win_size_sec`` default is ``'trial'``, meaning the trial duration will be used (this is not the same
        default as get_spa_ac_windowed, which is the function we are wrapping.)
        
        ``peak_fudge_cm`` is the width over which to take the np.mean to define the peak top and peak bottom.
        Note that for the peak top we use from 0-``peak_fudge_cm``, but for the peak bottom, once we've found
        the peak via S-G diff and zero-crossing, we use ``+-peak_fudge_cm/2``...bear in mind that the data is
        probably noisiest around zero because the zone has the smallest area.  The exact use of the cm value
        is a bit rough, meaning we are not that particular about rounding and offets etc. when indexing into bins.
        In fact we go one fudge further and actually lose the first bin entirely.
        
        You can provide an ``ac1d`` rather instead of getting one computed from scartch.
        
        TODO: check the indexing and stuff for the fitting (and other stuff).
        
        WARNING: there is definitely something not quite right with the indexing
        and even the S-G filter doesn't seem to be doing exactly what I want, only
        noticable for smaller scales.!!!!!!!
        """
        #raise NotImplementedError
        
        # convert some of the args from cms to bins        
        sg_w = np.ceil(smoothing/bin_size_cm)
        sg_w += 1 if sg_w % 2 == 0 else 0
        upper_bin = np.ceil(max_scale/bin_size_cm)
        lower_bin = np.floor(min_scale/bin_size_cm)
        peakFudge_bin = np.ceil(peak_fudge_cm/bin_size_cm)
        
        # get the unsmoothed 1d-displacement autocorrelogram
        if ac1d is None:
            ac1d = self.get_spa_ac_windowed(t, c, as_1d=True, smoothing_bins=0, 
                                            bin_size_cm=bin_size_cm,
                                            win_size_sec=win_size_sec,
                                            win_size_cm=win_size_cm,
                                            down_samp_rate=down_samp_rate)
        mean_rate = len(self.tet_times(t, c, as_type='x'))/self.duration/self.pos_samp_rate # mean_rate in units of spikes per pos-samp
        ac1d = ac1d[:int(upper_bin)].filled(0) / mean_rate #tidy up the np.array a little and normalise to np.mean rate
        
        peak_top = np.mean(ac1d[1:int(peakFudge_bin)]) # losing the first bin is a super fudge
        
        # get the first deriavative of the ac1d, using the Savitskay-golay method and find first down and first up zero-crossings
        # values were binned by rounding down, so bin centre is +0.5 from bin ind..actually see TODO note above
        dx = savgol_filter(ac1d, sg_w, sg_polyn, deriv=1, mode='mirror')      
        trough_ind, peak_ind = np.array(_find_threshold_crossing(dx[int(lower_bin):], 1, 1, val=0)) + lower_bin
        try:        
            before_ind = after_ind = np.nan
            before_ind, = _find_threshold_crossing(dx[:int(np.ceil(peak_ind)+1)], n_down=-1, val=thresh)
            after_ind, = _find_threshold_crossing(dx[int(np.floor(peak_ind)-1):], n_down=1, val=-thresh) + np.floor(peak_ind-1)
        except PointsOfInterestNotFound:
            pass
        peak_bottom = np.mean(ac1d[int(trough_ind)-int(np.ceil(peakFudge_bin/2)): np.int((trough_ind+np.ceil(peakFudge_bin/2)))])  # TODO: check this works as intended
        
        peak_top_sub = (peak_top-peak_bottom)*0.9 + peak_bottom
        peak_bottom_sub = (peak_top-peak_bottom)*0.1 + peak_bottom
        
        try:
            top_sub_ind = bottom_sub_ind = np.nan
            top_sub_ind, = _find_threshold_crossing(ac1d[:int(trough_ind)], n_down=-1, val=peak_top_sub)
            bottom_sub_ind, = _find_threshold_crossing(ac1d[:int(trough_ind)], n_down=1, val=peak_bottom_sub)
        except PointsOfInterestNotFound:
            pass                
        x_vals = np.arange(np.ceil(top_sub_ind), np.ceil(bottom_sub_ind))
        y_vals = ac1d[x_vals.astype(int)]
        coef = polyfit(x_vals/bin_size_cm, y_vals, deg=1)

        half_height = np.mean([peak_top, peak_bottom])
        half_width = (half_height-coef[0])/coef[1]
            
        # convert back to cms...the following points should be in order from centre outwards
        trough, before, scale, after = (np.array([ trough_ind, before_ind, peak_ind, after_ind])+0)*bin_size_cm
        top_sub, bottom_sub = (np.array([ top_sub_ind, bottom_sub_ind])+0)*bin_size_cm
                        
        clusteriness = half_width/scale
        peak_width = (after-before)/scale
        
        if return_extra is False:
            return scale, trough, clusteriness, peak_width
        else:
            locals_ = locals()
            return Info1dACProps(**{k: locals_[k] for k in Info1dACProps._fields})
       
    @append_docstring(AnalysisClass.get_spa_ac_windowed)
    def get_drift_score_1d(self, t, c, a_win=1200, b_win=20, 
                           return_extra=False, **kwargs):
        """
        Computes the 1d windowed sac for a window of length a and of length b.
        It then aligns the two to get the minimum squared error between them
        and returns the average sqred error as a proportion of the variance
        of the sac for length a.
        
        TODO: need to convert distance to bins and proeprly select out the
        range, currently hardcoded in bins.
        """
        raise DontTrustThisAnalysisException
        
        a, bin_size_cm = self.get_spa_ac_windowed(t, c, win_size_sec=a_win, as_1d=True,
                                          return_extra=True, **kwargs)
        b, _ = self.get_spa_ac_windowed(t, c, win_size_sec=b_win, as_1d=True,
                                          return_extra=True, **kwargs)
        a_sub = a[3:30]
        b_sub = b[3:30]
        best_offset = np.mean(a_sub) - np.mean(b_sub) # this is analytically the right thing to do to minimise sqred error
        
        sqrd_error = np.sum((a_sub-b_sub-best_offset)**2)/len(a_sub)
        normalizer = np.var(a_sub)

        if return_extra:
            return InfoDrift1d(a=a, b=b, bin_size_cm=bin_size_cm, var_a=normalizer, best_off=best_offset,
                            mean_diff=sqrd_error, score=sqrd_error/normalizer)
        else:
            return sqrd_error/normalizer
 

    @append_docstring(get_spa_ac_1d_props)
    def plot_spa_ac_1d_props(self, t=None, c=None, made_earlier=None, cla=True,
                             **kwargs):
        """
        plots the unsmoothed 1d autocorr in black, and the identified first peak in red.
        the blue line shows the S-G first derivative used for finding the peak.
        """
        if made_earlier is None:
            info = self.get_spa_ac_1d_props(t, c, return_extra=True, **kwargs)
        else:
            info = made_earlier
            
        if cla:
            plt.cla()
        
        DERIV_SCALE = 5
        plt.plot([0, 1], [info.peak_top, info.peak_top], 'c:', transform=trans_xaxes_ydata())        
        plt.plot([0, 1], [info.peak_bottom, info.peak_bottom], 'c:', transform=trans_xaxes_ydata())        

        plt.plot([0, 1], [0, 0], 'k', transform=trans_xaxes_ydata())        
        plt.plot([info.min_scale, info.min_scale], [0, 1], ':',
                 c=[0.7]*3, transform=trans_xdata_yaxes())
        plt.plot([0, 1], [info.thresh*DERIV_SCALE, info.thresh*DERIV_SCALE], ':',
                 c=[0.7]*3, transform=trans_xaxes_ydata())
        plt.plot([0, 1], [-info.thresh*DERIV_SCALE, -info.thresh*DERIV_SCALE], ':',
                 c=[0.7]*3, transform=trans_xaxes_ydata())
        plt.plot(np.arange(0, len(info.ac1d))*info.bin_size_cm+0.5*info.bin_size_cm,
                 info.ac1d, 'k', zorder=4)
        plt.plot([info.scale, info.scale], [0, 1], 'r', transform=trans_xdata_yaxes())
        plt.plot([info.trough, info.trough], [0, 1], 'g', transform=trans_xdata_yaxes())
        plt.plot([info.before, info.before], [0, 1], 'b', transform=trans_xdata_yaxes())
        plt.plot([info.after, info.after], [0, 1], 'b', transform=trans_xdata_yaxes())
        plt.plot(np.arange(0, len(info.ac1d))*info.bin_size_cm+0.5*info.bin_size_cm,
                 info.dx*DERIV_SCALE, c=[0.5, 0.5, 1])
        
        plt.plot([info.top_sub]*2, [0, 1], 'r:', transform=trans_xdata_yaxes())
        plt.plot([info.bottom_sub]*2, [0, 1], 'r:', transform=trans_xdata_yaxes())
        
        plt.plot([info.top_sub, info.bottom_sub],
                 [info.coef[0]+info.top_sub*info.coef[1],
                  info.coef[0]+info.bottom_sub*info.coef[1]],
                 'b', lw=2, zorder=0)      
        
        plt.plot([0, 1], [info.half_height]*2, 'c-', transform=trans_xaxes_ydata())
        plt.plot([info.half_width]*2, [0, 1], 'c-', transform=trans_xdata_yaxes())
                
        plt.title('scale={:0.1f}cm | C={:0.5f} | W={:0.0f}%'.format(
                    info.scale, info.clusteriness, info.peak_width*100),
                 fontsize=8)
    
    
def _find_threshold_crossing(x, n_up=0, n_down=0, val=0):
    """
    for a signal x, it returns the first ``n_up`` crossings from below ``val`` to above ``val``
    and the first ``n_down`` crossing from above ``val`` to below.
    the values are returned as a np.single tuple of up_0, up_2, ..up_n, down_0, down_1, ...down_m
    
    The values returned refer to the index within x, they are linearly interpolated.
    For example if ``x`` is ``[-1.1, -2.1, -0.1, 0.9, 1.9 ]``, the first upward-zero-crossing occurs between
    index 2 and 3, and in this case it is linearly interpolated to 2.1.

    ``n_up`` and/or ``n_down`` can be negative, in which case the last ``n_whatever`` values are returned not the first.
    TODO: make some attempt to handle edge cases..meaning actual edeges and zero values... and test properly.
    """

    downs_left_ind = ((x[:-1] >val) & (x[1:]<=val)).nonzero()[0]\
                                [slice(None, n_down) if n_down >= 0 else slice(n_down, None)]
    up_left_ind = ((x[:-1] <val) & (x[1:]>=val)).nonzero()[0]\
                                [slice(None, n_up) if n_up >= 0 else slice(n_up, None)]    
    left_inds = np.concatenate([up_left_ind , downs_left_ind])
    
    n_up, n_down = abs(n_up), abs(n_down)
    if len(left_inds) < n_up + n_down:
        raise PointsOfInterestNotFound("was looking for %d up-crossings and %d down-crossings, found %d in total." %(n_up, n_down, len(left_inds)))
        
    return tuple(left_inds + -(x[left_inds]-val) / (x[left_inds+1]- x[left_inds]))
