# -*- coding: utf-8 -*-

#from __future__ import division #TODO: update code to sensible division rules

import warnings

import numpy as np 
import scipy as sp

from numpy.fft import fft, ifft, fftshift
from numpy import newaxis as _ax

from scipy.ndimage import uniform_filter, binary_fill_holes
from scipy.ndimage.filters import gaussian_filter as sp_gaussian_filter
from scipy.spatial import Delaunay
from scipy.stats.mstats import gmean


from numpy_groupies import aggregate_np as aggregate
import numpy_groupies as npg

import skimage, skimage.morphology, skimage.measure, skimage.feature, skimage.segmentation

import mahotas

# If we use pyfftw instead of default nump.fft everything works the same but is slower unless 
# you actively optimise it. ...make that a TODO:
#from pyfftw.interfaces.numpy_fft import fft, ifft, fftshift 
from collections import namedtuple

from utils.custom_itertools import (batched)
from utils.custom_casting import str2int
from utils.custom_functools import append_docstring
from utils.custom_exceptions import DontTrustThisAnalysisException

from utils.utils_circle_fit import fit_circle
from utils.utils_ellipse_fit import Ellipse

from noncore import dist_to_peak
from noncore import run_peak_rates

from core.trial_basic_analysis import TrialBasicAnalysis as AnalysisClass # used only for docstrings


warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")


Circle = namedtuple('Circle', "xc yc r")

InfoSpaStability = namedtuple('InfoSpaStability', 'a b r p')
InfoSpaAcPeakFit = namedtuple('InfoSpaAcPeakFit', ("best_C best_orient best_scale "
                                         "best_phase ac n bin_size_cm center coords"))
InfoRateLL = namedtuple('InfoRateLL', "LL_total pos_likelihood")
InfoRunPeaks = namedtuple('InfoRunPeaks', ("height_mean height_sem width_mean "
                                         "width_sem"))

InfoGCFields = namedtuple('InfoGCFields', ("labels bin_size_cm field_area rm "
                                           "peak_bin_inds field_area_representative"))
InfoFieldProps = namedtuple('InfoFieldProps', ("tri tri_used tri_scale circs "
                                         "field_area_median field_area_mean "
                                         "field_area_geomean boundary_labels "
                                         "peak_xy"))
InfoGridness = namedtuple("InfoGridness", ("used_ellipse ellipse ac_nowarp "
                                         "field_perim scale scale_list orientation "
                                         "mean_r_in_label gridness gridness_mask "
                                         "ac_masked rotational_corr_values "
                                         "closest_peaks_coord peaks_label "
                                         "fields_label ac bin_size_cm central_area"))


                        
class TrialGCAnalysis(object):
    """ This is a mixin class.
    TODO: some of this stuff could be pulled out into separate mixins"""
     
    @append_docstring(AnalysisClass.get_spa_ratemap)
    def get_spa_ac(self, t=None, c=None, rm=None, return_n=False,
                   normalise_n=False, **kwargs):
        """
        You can either pass a ratemap or pass the parameters needed for building
        a ratemap. ``t``, ``c``, and ``bin_size_cm``, and ``smoothing``.
        
        ``return_n`` gets pased through to _spa_ac_from_rm function. If true
        the output will be (ac, n), bin_size_cm, otherwise it will just be ac, bin_size_cm.
        """
        if rm is None:
            rm, bin_size_cm = self.get_spa_ratemap(t=t, c=c, **kwargs)
        elif 'bin_size_cm' in kwargs:
            bin_size_cm = kwargs['bin_size_cm']
        else:
            bin_size_cm = None
            
        return _spa_ac_from_rm(rm.filled(fill_value=0), rm.mask, 
                               nodwell_is_zeroed=True, return_n=return_n,
                               normalise_n=normalise_n), bin_size_cm
              
        
    def get_gc_fields(self, t, c,
                    bin_size_cm=2.5, mode=1, min_area_cm2=40,
                    min_area_relative=(2, 0.6),  pos_mask=None,
                    m1_smoothing_sigma_cm=5, m1_thresh_peak_frac=0.5,
                    m1_smooth_before=True, m2_smoothing_type='adaptive', 
                    m2_smoothing=2500, m2_field_pthresh=75, m3_field_pthresh=50,
                    as_boundaries=True, as_masked=True, return_extra=False,
                    representative_area_k=2):
        """
        There are two modes:
        
        **Mode 1**    
        Builds a non-smoothed ratemap, then smooths it loads with Guassian smoothing.
        Then finds peaks, watersheds to get zone around each peak. Then thresholds
        each zone at half peak height to get fields.  This is the method that was 
        developed as part of the Jeewajee et al. 2014 (though I think an older
        algorithm was described in the paper).  It is partly derived from a method
        used in the gridness scoring algorithm.
        
        **Mode 2**   
        Build an adaptivly smoothed ratemap.  Threshold at 75th percentile to get fields.
        This is the method used in Hardcastle et al. 2015.
        
        **Mode 3**
        Build a gaussian smoothed ratemap, then threshold based on nanpercentile (50)
        This is roughly what Caswell does. ``m3_field_pthresh`` can be a percentile,
        1 to 100, or the string ``"np.mean"``.
        
        Can return a 2d np.array of either either the labeled fields, or labeled boundaries.
        If as_masked is true the output np.array will be a numpy masked np.array, 0/False set
        to masked. If return_extra is True various other bits and bobs will be returned.
        
        Any fields with area less than `min_area_cm2` will be removed.
        Alternatively, you can specify a relative threshold, using ``min_area_relative``.
        It should be a tuple, the threshold is defined by sorting the areas, from largest
        to smallest, taking the ``min_area_relative[0]``th element (i.e. 0=largest) and
        multiplying by ``min_area_relative[1]``, i.e. a value on [0, 1].
        ``min_area_cm2`` will be applied in addition (i.e. threshold will be at least this
        value, so set to 0 to ignore it).
        
        representative_area_k - the field_area_representative value is the kth largest
        field's area, or nan if not enough fields are present.  k is 0-based, i.e
        k=1 is the second largest field's area. Can also be None to avoid analysis.
        
        """
        peak_bin_inds = None # not all modes actually find peaks
            
        if mode == 1:
            
            if m1_smooth_before:
                # Get unsmoothed ratemap, with nodwell bins set to 0.
                rm, _ = self.get_spa_ratemap(t=t, c=c, bin_size_cm=bin_size_cm,
                                smoothing_type='gaussian', 
                                smoothing_bins=m1_smoothing_sigma_cm/bin_size_cm,
                                nodwell_mode=0, pos_mask=pos_mask)                
            else:
                # Get unsmoothed ratemap, with nodwell bins set to 0.
                rm, _ = self.get_spa_ratemap(t=t, c=c, bin_size_cm=bin_size_cm,
                                smoothing_bins=0, nodwell_mode=0,
                                pos_mask=pos_mask)                
                
                # Now smooth (normally we would smooth dwell and nodwell before division)
                rm = sp_gaussian_filter(rm, m1_smoothing_sigma_cm/bin_size_cm,
                                        mode='nearest')
            
            # Reduce bitdepth to 8
            m = np.amax(rm)
            if m == 0:
                rm8 = np.zeros(rm.shape, dtype=np.uint8)
            else:
                rm8 = (rm * 255./m).astype(np.uint8)
                   
            # Find the peaks and watersheds around each peak to get labels
            peak_labels, _ = _im_regional_max(rm8, min_peak_height=10)
            field_labels = _find_peak_extent_multi(rm, peak_labels,
                                                thresh_frac=m1_thresh_peak_frac)
            if return_extra:
                peak_bin_inds = peak_labels.nonzero()
                peak_labels = field_labels[peak_bin_inds]
                peak_bin_inds = np.hstack(( [[-1], [-1]], np.asarray(peak_bin_inds))) #-1s for zero label
                peak_bin_inds[:, peak_labels] = peak_bin_inds[:, 1:]
            
        elif mode == 2:
            # Get adpatively smoothed ratemap
            rm, _ = self.get_spa_ratemap(t, c, smoothing_type=m2_smoothing_type,
                                  smoothing_bins=m2_smoothing,
                                  bin_size_cm=bin_size_cm, pos_mask=pos_mask)                          

            # threshold and label      
            thresh = np.percentile(rm[~rm.mask], m2_field_pthresh)            
            field_mask = binary_fill_holes(rm.filled(0) > thresh)
            field_labels = skimage.measure.label(field_mask, 8)
            field_labels[~field_mask] = 0 # it's possible the non-peak area is split up by the peaks

            if return_extra:
                # get max bin in each field
                peak_bin_inds = aggregate(field_labels.ravel(), rm.ravel(), func='argmax')
                peak_bin_inds = np.asarray(np.unravel_index(peak_bin_inds, rm.shape))
                
        elif mode == 3:
            # threshold relative to global rate
            rm, _ = self.get_spa_ratemap(t, c, bin_size_cm=bin_size_cm,
                                        smoothing_bins=2, 
                                        smoothing_type='gaussian',
                                        nodwell_mode='ma', pos_mask=pos_mask)  
            if m3_field_pthresh == "np.mean":
                thresh = np.mean(rm.compressed())
            else:
                thresh = np.percentile(rm.compressed(), m3_field_pthresh)            
            field_mask = binary_fill_holes(rm.filled(0) > thresh)
            field_labels = skimage.measure.label(field_mask, 8)
            field_labels[~field_mask] = 0 # it's possible the non-peak area is split up by the peaks
            
            
        field_area = None
        if min_area_cm2 > 0 or min_area_relative is not None or representative_area_k is not None:
            # remove fields that are smaller than min_area_cm2
            # and re-label remaining fields from 1...n
            field_area = aggregate(field_labels.ravel(), bin_size_cm**2)
            field_area_sorted = np.sort(field_area[1:])
            if representative_area_k is not None:
                if len(field_area_sorted) > representative_area_k:
                    field_area_representative = field_area_sorted[representative_area_k]
                else:
                    field_area_representative = np.nan
                
            if min_area_relative is not None:
                min_area_cm2 = max(min_area_cm2, 
                                   field_area_sorted[min(min_area_relative[0], len(field_area)-2)]\
                                           *min_area_relative[1])
            keep_field = field_area > min_area_cm2
            field_area = field_area[keep_field] # update this for what will be the new labels
            keep_field[0] = False
            new_label = np.cumsum(keep_field)
            new_label[~keep_field] = 0
            field_labels = new_label[field_labels]
            if peak_bin_inds is not None:
                keep_field[0] = True # peak_bin 0 is [-1, -1] for clarity, so keep it
                peak_bin_inds = peak_bin_inds[:, keep_field]
        ret = field_labels        

        if as_boundaries is True:
            # zero the interior of the fields
            ret[~_boundary_mask(ret)] = 0

        # if we know the shape of the environment we should crop out anything outside the environment
        try: 
            environment_mask = self.pos_shape.make_mask(self.w, self.h, bin_size_cm)
        except AttributeError:
            environment_mask = None
        if environment_mask is not None:
            environment_mask = skimage.morphology.binary_dilation(environment_mask,
                                                                  np.ones((3, 3))).astype(bool) # this is a bit of a nasty hack to supress stuff from just inside the boundarys
            ret[environment_mask] = 0
            
        if as_masked:
            ret = np.ma.array(ret, mask=ret==0)
        
        if return_extra is False:
            return ret
        else:
            return InfoGCFields(labels=ret,
                                bin_size_cm=bin_size_cm,
                                field_area=field_area,
                                rm=rm,
                                peak_bin_inds=peak_bin_inds,
                                field_area_representative=field_area_representative)


    @append_docstring(get_gc_fields)
    def get_field_label_at_xy(self, t=None, c=None, mode='spike', mode_fields=None, **kwargs):
        """for each pos it reads the relevant pixel in the fields labels map.
        Similarish to _getRateAtXY in terms of indexing into a ratemap shaped thing.        
       
        ``kwargs`` passed to ``get_gc_fields``.
        """
        if mode_fields is not None:
            kwargs['mode'] = mode_fields # neccessary because of a clash of kwarg names: we want to use mode here for output type
            
        info = self.get_gc_fields(t, c, as_boundaries=False, as_masked=False,
                                  return_extra=True, **kwargs)
        
        xy_idx, _, _ = self.xy_bin_inds(info.bin_size_cm)
        
        if mode == 'pos':
            return info.labels[xy_idx[1, :], xy_idx[0, :]]
        elif mode == 'spike':
            return info.labels[xy_idx[1, info.spk_pos_inds],
                               xy_idx[0, info.spk_pos_inds]]
        else:
            raise Exception("unknown mode")
            
    def get_gc_measures(self, t=None, c=None, ac=None,
                      field_extent_method=2, exclude_props=('central_area',), 
                      orientation_mode='mean',  ellipse_transform=False,                    
                      **kwargs):
        '''
        A rough clone of the Matlab file autoCorrProps.m on GitHu
        
        ``orientation_mode`` can be ``"mean"`` or ``"first"``, indicating whether
        to use mean of all 3 peaks or jus the first anticlockwise from East.
        
        If ``ellipse_transform`` is ``True``, we fit an ellipse to the central
        peaks and then apply a transform to the sac to turn the ellipse into a
        circle, before doing the gridness calculation.
        '''
        
        if ac is None: #you can pass in a ready-made autocorr if you like
            ac, bin_size_cm = self.get_spa_ac(t, c, **kwargs)
        else:
            bin_size_cm = kwargs.get('bin_size_cm', None)
        
        # [STAGE 1] find peaks & identify 7 closest to centre
        peaks_label, xy_coord_peaks = _im_regional_max(ac)
        xy_coord_peaks = xy_coord_peaks.transpose()

        # Convert to a new reference frame which has the origin at the centre of the autocorr
        central_point = np.floor(np.array(ac.shape)/2).astype(int)  # fixed: this should be floor not ceil
        xy_coord_peaks_central = xy_coord_peaks - np.reshape(central_point, (1, 2))
        
        # calculate distance of peaks from centre and find 7 closest (closest is actually the central peak)
        peaks_dist_to_centre = np.hypot(xy_coord_peaks_central[:, 1], xy_coord_peaks_central[:, 0])
        order_of_close = np.argsort(peaks_dist_to_centre)
        closest_peaks = order_of_close[:7] # might be less than 7 (if so numpy will just return the full np.array)
        closest_dist_to_centre = peaks_dist_to_centre[closest_peaks]
        xy_coord_peaks = xy_coord_peaks
        n_peaks = len(closest_peaks)
        used_ellipse = False 
        ellipse = None
        
        if ellipse_transform and n_peaks == 7:
            # get matrix for transforming ellipse of central peaks into a cricle
            # and apply to autocorr and to peak coords, and update peak dists
            ellipse = Ellipse.from_xy(*xy_coord_peaks[closest_peaks[1:], ::-1].T.astype(float))            
            if ellipse.is_ok():
                used_ellipse = True
                ac_nowarp = ac
                ac = skimage.transform.warp(ac.filled(0), 
                                           inverse_map=ellipse.from_circle_transform_matrix,
                                           preserve_range=True)
                ac = np.ma.array(ac)
                xy_coord_peaks_agumented = np.hstack((xy_coord_peaks[:, ::-1],
                                            np.ones(len(xy_coord_peaks))[:, _ax]))
                xy_coord_peaks = np.dot(ellipse.to_circle_transform_matrix,
                                      xy_coord_peaks_agumented.T)
                xy_coord_peaks = np.asarray(np.rint(xy_coord_peaks[1::-1, :].T).astype(int))
                xy_coord_peaks[:, 0] = np.clip(xy_coord_peaks[:, 0], 0, ac.shape[0]-1)
                xy_coord_peaks[:, 1] = np.clip(xy_coord_peaks[:, 1], 0, ac.shape[1]-1)
                closest_dist_to_centre = np.hypot(xy_coord_peaks[:, 1], xy_coord_peaks[:, 0])
                xy_coord_peaks_central = xy_coord_peaks - np.reshape(central_point, (1, 2))
        closest_peaks_coord = xy_coord_peaks[closest_peaks, :]
        closest_coord_peaks_central = xy_coord_peaks_central[closest_peaks, :]
        
        # [Stage 2] Expand peak pixels into the surrounding half-height region
        if field_extent_method == 1:
            fields_label = np.zeros(ac.shape, dtype=int)
            for i in range(n_peaks):
                fields_label[_find_peak_extent(ac, closest_peaks_coord[i, :])] = i+1
        elif field_extent_method == 2:
            # here we use watershed, for which we need to provide all the peaks
            # or else all points will be assigned to one of the central peaks.
            fields_label = _find_peak_extent_multi(ac, xy_coord_peaks[order_of_close, :]) 
            fields_label[fields_label>7] = 0
            
        fields_mask = fields_label.astype(bool)            
        field_perim = mahotas.bwperim(fields_mask)
            
        if any(p not in exclude_props for p in ['scale', 'mean_r_in_label', 'orientation']):
            # [Stage 3] Calculate a couple of metrics based on the closest peaks
            # Find the (np.mean) autoCorr value at the closest peak pixels
            mean_r_in_label = aggregate(fields_label[fields_mask]-1, 
                                        ac[fields_mask], func='mean', size=n_peaks)
            scale = np.median(closest_dist_to_centre[1:])*bin_size_cm
            scale_list = closest_dist_to_centre[1:]*bin_size_cm
            orientation = _get_orientation(closest_coord_peaks_central[1:, :], orientation_mode)
            
        if 'central_area' not in exclude_props:
            # get area of central peak in cm (thresholded at zero)            
            central_area = np.count_nonzero(_find_peak_extent(ac,
                                                              central_point,
                                                             thresh_abs=0)) * bin_size_cm**2
            
        # [Stage 4] Calculate gridness
        if 'gridness' not in exclude_props:
            # Find the max distance from the edge of one of the 6 peaks to the centre
            xx = np.reshape(np.arange(-central_point[0], central_point[0]+1), (-1, 1))
            yy = np.reshape(np.arange(-central_point[1], central_point[1]+1), (1, -1))
            dist_to_centre = np.hypot(xx, yy)
            max_dist_from_center = np.nan
            if len(closest_peaks) >= 7:
                max_dist_from_center = max(dist_to_centre[fields_mask])
            if np.isnan(max_dist_from_center) or max_dist_from_center > min(central_point):
                max_dist_from_center = min(central_point)
              
            #create the gridness_mask, which is a disk of radius max_dist_from_center, 
            #with the central field cut out
            gridness_mask_all = dist_to_centre <= max_dist_from_center
            centreMask = fields_label == fields_label[central_point[0], central_point[1]]
            gridness_mask = gridness_mask_all & ~centreMask
            
            # Apply the gridenss mask to get a the autoCorr middle
            W = np.ceil(max_dist_from_center).astype(int)
            ac_masked = ac.copy()
            ac_masked[~gridness_mask] = np.ma.masked
            ac_centre = ac_masked[-W + central_point[0]:W + central_point[0], -W+central_point[1]:W+central_point[1]]
            
            # Calculate the corelations of the autocormiddle with rotated versions of itself
            rotational_corr_values = _get_rotational_corrs(ac_centre, 
                                                           [30, 60, 90, 120, 150])
            
            # Compue gridness
            gridness = min((rotational_corr_values[60], rotational_corr_values[120])) - \
                            max((rotational_corr_values[150], rotational_corr_values[30], rotational_corr_values[90]))    
                    
        locals_ = locals()
        return InfoGridness(**{k: locals_.get(k, None) 
                               for k in InfoGridness._fields})
    
    @append_docstring(AnalysisClass.get_spa_ratemap)
    def get_gridness_and_shuffle(self, t, c, min_shift_s=20,
                                 n_shuffles=100, **kwargs):
        """
        TODO: this is unfinished.
        """
        
        # Generate an np.array of n_shuffles of temporal offsets spread uniformly along trial length,
        # but with a window of minShiftS either side of 0 (times are going to be wrapped around)
        shift = np.random.uniform(min_shift_s, self.duration-min_shift_s, n_shuffles+1) 
        shift[0] = 0 # the first element is going to be unshifted to give us the true gridness..that's why +1 above
        
        # compute all the ratemaps
        rm, bin_size_cm = self.get_spa_ratemap(t, c, spike_shifts_s=shift, 
                                               nodwell_mode='nan', **kwargs)        
        ac = _spa_ac_from_rm(rm, np.isnan(rm[..., 0]), nodwell_is_zeroed=False)
        ac = np.rollaxis(ac, 2) # now ac[ii] is the ii'th autocorr
        
        true_gridness_info = self.get_gc_measures(ac=ac[0], 
                                                  bin_size_cm=bin_size_cm) # remeber 0th shuffle has no shift
        shuffled_gridness = np.empty(n_shuffles)
        for ii, ac_ii in enumerate(ac[1:]):
            shuffled_gridness[ii] = self.get_gc_measures(ac=ac_ii,
                                                         bin_size_cm=bin_size_cm,
                                                         all_props=False).gridness
            
        shuffled_gridness.sort()
        return true_gridness_info, shuffled_gridness
        
    @append_docstring(dist_to_peak.fit_peaks)
    def get_spa_ac_peak_fit(self, t=None, c=None, w_thresh=0.05,
                     scales_linspace=(30, 55, 30), phase_nish=None, orient_n=30,
                    return_extra=False, **kwargs):
        """
        Based on Yoon et al. 2013. but not identical.
        By DM, oct 2014.
        
        Notably, we multiply the Yoon weights by n, the number of overlapping bins in that
        bit of the autocorrelogram.  Basically, this means that going out from the center 
        the weights tend to zero, so rubbish little peaks don't contribute much.

        Then, for the sake of faster computation, we discard the smallest peaks, which
        only contribute to the final 5% of the total weighting.  (``w_thresh=0.05`` is
        the 5%, you can change it to soemthing else if you want.)
        
        Note that fitting is done in bins, though plot title shows result in cm.
        """
        
        rm_kwargs = {'smoothing_type':'gaussian',
                    'smoothing_bins':2,
                    'bin_size_cm':1}
        rm_kwargs.update(**kwargs)
        
        # get the 2d autocorr, together with the number of overlapping bins for each correlation
        (ac, n), bin_size_cm = self.get_spa_ac(t, c, return_n=True, normalise_n=True, **rm_kwargs)

        # Find the peaks and their weights
        peaks, coords = _im_regional_max(ac)        
        w = ac[coords[0], coords[1]] * n[coords[0], coords[1]]     
        
        # Discard the peaks contributing to the tail of the weights distribution
        w_ix = np.argsort(w)[::-1]
        w_sorted_within_thresh = np.cumsum(w[w_ix]) > sum(w) *(1-w_thresh)
        good_ix = w_ix[:np.searchsorted(w_sorted_within_thresh, True)+1]
        w = w[good_ix]
        coords = coords[:, good_ix]
        
        # Note we shift we shift the coord values, so that the centre of the ac is (0, 0).
        # I don't think this is critical, but it should make life easier.
        # TODO: it's stupid having a full rnage of phases for autocorellograms..should limit to (0, 0) or close
        center = (int(n.shape[0]/2)+1, int(n.shape[1]/2)+1)
        best_C, best_orient, best_scale, best_phase = dist_to_peak.fit_peaks(
                                    coords.T-np.array(center)[_ax, :], w, 
                                    scales=np.linspace(*scales_linspace),
                                    phase_nish=phase_nish, orient_n=orient_n)
        if return_extra is False:
            return InfoSpaAcPeakFit(best_C=best_C,
                               best_orient=best_orient,
                               best_scale=best_scale,
                               best_phase=best_phase,
                               ac=None, n=None, bin_size_cm=None, center=None, coords=None)

        else:
            return InfoSpaAcPeakFit(best_C=best_C,
                               best_orient=best_orient,
                               best_scale=best_scale,
                               best_phase=best_phase,
                               ac=ac,
                               n=n,
                               bin_size_cm=bin_size_cm,
                               center=center,
                               coords=coords)
        
        
        
    def get_pos_is_in_field(self, t, c, as_type='p'):
        """ Uses get_gc_fields to determine for each pos sample whether it lies in the field or not
        """
        info = self.get_gc_fields(t, c, as_boundaries=False, as_masked=False, return_extra=True)
        xy_idx , _, _= self.xy_bin_inds(bin_size_cm=info.bin_size_cm)
        posIsInField = info.labels[[xy_idx[1, :], xy_idx[0, :]]].astype(bool)
        if as_type == 'p': #in pos samp rate
            return posIsInField
        elif as_type == 'e': #in eeg samp rate
            upSamp = str2int(self.eegHeader['EEG_samples_per_position'])
            ret = np.repeat(np.reshape(posIsInField, [-1, 1]), upSamp, axis=1)
            return ret.ravel()
            
    def get_gravity_transform(self, t, c, norm='rank', sum_mode='complex',
                            w_cm=16, nodwell_mode='ma', **kwargs):
        """
        For each bin the ratemap it computes ``sum_b(r_a*r_b/d2(a, b))``, where ``r_a``
        is the rate for the bin in question and ``r_b`` is the rate of another bin
        at distance-squared ``d2(a, b)``.  In otherwords we are computing the total
        magintude of "gravitational"/"coloumbic" force actign on the bin.
        ``w_cm`` controls the size of the kernel - there is no gaussian-like width, only
        a threshold, beyond which it's safe to treat the inverse d2 as being
        essentially zero. There should be no real need to change this value.
        The ratemap we use is controlled by ``kwargs``, but norm is set to `rank`.
        
        Note that the gravity transform is esentially a kind of smoothing (once
        you've done the ranking step.)
        
        The above description si for ``sum_mode='simple'``, for ``sum_mode='complex'``
        we compute the actual net force vector acting on each bin rather than the
        semi-nonsensical, simple sum of components.  This produces a complex output.
        
        WARNING: I don't think this function is actually good for anything.
        """
        
        rm, bin_size_cm = self.get_spa_ratemap(t, c, norm='rank',
                                               nodwell_mode='nan', **kwargs)
        mask = np.isnan(rm)
        rm[mask] = 0
        
        W = np.ceil(w_cm/bin_size_cm)
        kern = (np.arange(-W, W+1)[:, _ax]**2 + np.arange(-W, W+1)[_ax, :]**2) * bin_size_cm**2 
        if sum_mode == 'simple':
            kern = 1.0/kern
        elif sum_mode == 'complex':
            kern = kern**-1.5 * (np.arange(-W, W+1)[:, _ax] + 1j*np.arange(-W, W+1)[_ax, :])
        else:
            raise Exception("what?")
        kern[int(W), int(W)] = 0
        G = sp.signal.convolve2d(rm, kern, mode='same')*rm
        if nodwell_mode == 'ma':
            return np.ma.array(G, mask=mask)
        elif nodwell_mode == 0:
            return G
        elif nodwell_mode == 'na':
            G[mask] = np.nan
            return G
        else:
            raise Exception("what?")

    def get_spa_stability(self, t, c, return_extra=False,
                        smoothing_type='adaptive', smoothing=2500,
                        bin_size_cm=1.4,
                        region_labels=None,
                        drop_common_thresh=None,
                        compare_mode=1,
                        **kwargs):
        """
        Compute normal ratemap, but for data from first half of trial only.  Then
        do pearson correlation on the bins that are non-np.nan in both (i.e. had some
        dwell in both parts of trial)
        
        If ``region_labels`` is not None, it should be an np.array the same size as
        the ratemaps that will be produced here.  The stability within each
        non-zero labeled section of the evironent will be computed and returned
        in a vector, with the 0th element of the vector giving the overal stability
        and the kth element giving the stability of the kth labeled region.
        
        ``drop_common_thresh`` - if this is not ``None``, then any areas which
        has a value less than or equal to this in both ratemaps will be dropped, 
        i.e. if you set it to zero then any bins with zero in both ratemaps will
        be dropped.
        
        ``compare_mode=1``  means split trial in half, ``compare_mode=2`` means
        split trial into 4 and compare [0 with 2] and [1 with 3], returning the
        np.mean correlation as the stability value.
        """      
        kwargs.update(smoothing_type=smoothing_type,
                      smoothing_bins=kwargs.get('smoothing_bins', smoothing),
                      bin_size_cm=bin_size_cm)
        xy_full = self.xy
        spk_pos_inds_full = self.tet_times(t=t, c=c, as_type='p')
        
        if compare_mode == 1:
            split_pos = [int(self.n_pos/2)]
        else:
            split_pos = [int(self.n_pos/4), int(self.n_pos/2), int(self.n_pos*3./4)]
            
        xy_split = [xy_full[:, slice(from_, to_)] for from_, to_ in
                    zip([None] + split_pos, split_pos + [None])]
        split_spk_ind = list(np.searchsorted(spk_pos_inds_full, split_pos))
        spk_pos_inds_split = [spk_pos_inds_full[slice(from_, to_)] - off 
                              for from_, to_, off in 
                              zip([None] + split_spk_ind, split_spk_ind + [None], [0] + split_pos)]
        rm = [self.get_spa_ratemap(xy=xy_sub, spk_pos_idx=spk_pos_inds_sub, **kwargs)[0]
              for xy_sub, spk_pos_inds_sub in zip(xy_split, spk_pos_inds_split)]
        
        validMask = reduce(np.logical_and, [~x.mask for x in rm])
        if drop_common_thresh is not None:
            sub_thresh =  reduce(np.logical_and, [x.filled(np.Inf) <= drop_common_thresh
                                                  for x in rm])
            validMask = validMask & ~sub_thresh
            
        
        if compare_mode == 1:
            def stab_func(rm_list, rm_valid_mask):
                return sp.stats.pearsonr(rm_list[0][rm_valid_mask], rm_list[1][rm_valid_mask])
        elif compare_mode == 2:
            def stab_func(rm_list, rm_valid_mask):
                r_1, p_1 = sp.stats.pearsonr(rm_list[0][rm_valid_mask], rm_list[2][rm_valid_mask])
                r_2, p_2 = sp.stats.pearsonr(rm_list[1][rm_valid_mask], rm_list[3][rm_valid_mask])
                return np.mean([r_1, r_2]), np.max([p_1, p_2])
                    
        r, p = stab_func(rm, validMask)
        # Note that p-value is 2-tailed, which maybe isn't what you want here,
        # but could divide by 2 if needed.
        
        if region_labels is not None:
            if region_labels.shape != rm[0].shape:
                raise Exception("region_labels doesnt match shape of ratemaps")
            # for each of the k labeled regions, do a correlation just on that area
            k_vals = np.unique(region_labels[region_labels.astype(bool)])
            r = np.hstack((r, np.zeros(k_vals[-1])+np.nan))
            p = np.hstack((p, np.zeros(k_vals[-1])+np.nan))
            for k in k_vals:
                k_mask = region_labels==k
                r[k], p[k] = stab_func(rm, validMask & k_mask)
                
        if return_extra is True:
            if drop_common_thresh:
                for x in rm:
                    x[sub_thresh] = np.ma.masked
            return InfoSpaStability(a=rm[0], b=rm[1], r=r, p=p) # TODO: support other compare_modes
        else:
            return r, p
        

    @append_docstring(run_peak_rates.get_pos_peak_info)
    def get_pos_peak_info(self, t, c, **kwargs):
        """
        See also ``get_run_peak_info``. This doesn't segment on runs, it just finds 
        peak widths in smoothed rate. Peaks widths are in cm as with get_run_peak_info,
        as with that function, they are defined as the maximum distance between
        points on the pos segment for the half-peak height section.
        """
        return run_peak_rates.get_pos_peak_info(self, t, c, **kwargs)
        

    @append_docstring(run_peak_rates.get_run_peak_info)     
    def get_run_peak_info(self, t, c, test_mode=False, **kwargs):
        """
        Uses run_peak_rates.get_run_peak_info to get the peak heights and
        widths for fitlered selection of runs.  Here we report simply the
        np.mean and sem of the peak height and width.
        """        
        info = run_peak_rates.get_run_peak_info(self, t, c, return_extra=test_mode, **kwargs)
    
        if not test_mode:
            return InfoRunPeaks(height_mean=np.mean(info.runC_peak_rate),
                                 height_sem=sp.stats.sem(info.runC_peak_rate),                             
                                 width_mean=np.mean(info.runC_peak_width),                             
                                 width_sem=sp.stats.sem(info.runC_peak_width),                             
                                 ) 
        else:
            return InfoRunPeaks(height_mean=np.mean(info.runC_peak_rate),
                                 height_sem=sp.stats.sem(info.runC_peak_rate),                             
                                 width_mean=np.nanmean(info.runB_run_width),                             
                                 width_sem=sp.stats.sem(info.runB_run_width[~np.isnan(info.runB_run_width)]),                             
                                 ) 
    
    def get_spa_coherence(self, t=None, c=None, rm=None, **kwargs):
        """
        TODO: this has not exactly been tested.
        Calculates coherence of a ratemap according to Muller & Kubie 1989.
        
        You can provide a ratemap as `rm` or use `t` and `c` to get one from scratch
        but WARNING: you probably only want to be using this measure with unsmoothed rms.
        """
        if rm is None:
            rm, bin_size_cm = self.get_spa_ratemap(t=t, c=c, smoothing_bins=0, **kwargs)
            
        return _coherence(rm.filled(0), rm.mask, nodwell_is_zeroed=True)       
        
    def get_ratemap_compression_ratio(self, t, c, quality=25, format_='jpeg',
                                      **kwargs):
        """
        This does do something, but it doesn't seem very meaningful. Possibly 
        ends up being roughly equivalent to the ratio of the peak to the np.mean.
        """
        raise DontTrustThisAnalysisException
        
        from io import BytesIO
        from PIL import Image        
        rm, _ = self.get_spa_ratemap(t, c, **kwargs)
        b = BytesIO()
        img = Image.fromarray((rm.filled(0)/np.max(rm)*255).astype(np.uint8))
        img.save(b, format_, quality=quality)
        return b.tell()/float(len(rm.ravel()))
        
    def get_gc_field_measures(self, t, c, angle_tol_deg=15, area_thresh=(2, 0.6),
                           get_circs=True, get_tris=True, get_area=True,
                           field_kwargs={}):
        """
        Uses calls get_gc_fields and then calculates a bunch of stuff:

        ``circs`` - if ``get_circs`` is True.  Fits a circle to the perimiter
                    bins for each field.
                    
        ``tri``, ``tri_scale``, ``tri_used``` - if ``get_tris`` is True. Uses 
        delaunay triangulation of peaks, and some crazy filtering of the 
        triangles to get a hopefully robust esitmate of scale from peaks.
        ``area_thresh`` is tuple, first element gives the number k, for the kth
        largest triangle, where k=1 is the largest (k=0 is not valid). the second
        elemnt in the tuple gives the fraction of the kth tri's area to use
        as threshold.  ``tri_used`` says which triangles made it though the filter.
        See plotRatemapWithGCFieldMeasures for useage of these outputs.
        
        ``field_area_*`` - if ``get_area`` is True. np.mean, meadian and geomean, taken over all the 
        fields output from get_gc_fields.
        
        todo:
        * var of peak rate
        * var of field size
        * var of nearest neighbour dists
        * circularity of fields

        """
        
        boundary_labels, bin_size_cm, field_areas, _, peak_bin_inds, _ = \
                    self.get_gc_fields(t, c, as_boundaries=True, return_extra=True, **field_kwargs)
        
        peak_xy = tri_used = tri_scale =tri = field_area_median = field_area_mean = field_area_geomean = circs =None
        #=============
    
        if get_tris:
            peak_xy = peak_bin_inds * bin_size_cm
            points = peak_xy.T[1:]
            if len(points) > 2:
                tri = Delaunay(points)
        
                # get length of sides of all triangles
                c0 = points[tri.simplices, 0].astype(float)
                c1 = points[tri.simplices, 1].astype(float)
                # 0 1 2
                # a = dist(0, 1)  b=dist(0, 2) c=dist(1, 2)
                a = np.hypot(c0[:, 0]-c0[:, 1], c1[:, 0]-c1[:, 1])
                b = np.hypot(c0[:, 0]-c0[:, 2], c1[:, 0]-c1[:, 2])
                c = np.hypot(c0[:, 2]-c0[:, 1], c1[:, 2]-c1[:, 1])
            
                # use side lengths to get angles (sine rule)
                ang_0 = np.arccos((a**2+b**2-c**2)/(2*a*b))
                b, c = c, b
                ang_1 = np.arccos((a**2+b**2-c**2)/(2*a*b))
                ang_2 = np.pi - ang_0-ang_1
                ang = np.vstack((ang_0, ang_1, ang_2))
                
                # drop triangles with any angles outside 60+-angle_to_deg
                bad = np.any((ang < (60.-angle_tol_deg)/180*np.pi) | (ang > (60.+angle_tol_deg)/180*np.pi), axis=0)
                a, b, c = a[~bad], b[~bad], c[~bad]
        
                # calculate areas of remaining triangles (heron's formula)     
                s = (a+b+c)/2
                area = np.sqrt(s*(s-a)*(s-b)*(s-c))
                side_len = np.mean(np.vstack((a, b, c)), axis=0)
            
                if len(area) > 0:
                    # get scale from remainging triangles, that are considered large enough
                    k = max(len(area) - area_thresh[0], 0) # np.partition puts kth element at it's location as if sorting the whole np.array in ascending order
                    area_k = np.partition(area, k)[k] # find the kth largest area
                    bad_sub = area < area_thresh[1]*area_k
                    tri_scale = np.median(side_len[~bad_sub])    
                    bad[~bad] = bad_sub
                else:
                    bad[:] = True
                tri_used = ~bad
    
        #=============

        nFields = np.amax(boundary_labels)
        if get_circs:                  
            circs = []
            for f in range(nFields):
                x, y = (boundary_labels==f+1).nonzero()
                xc, yc, r = fit_circle(y+0.5, x+0.5) #TODO: check that the +0.5 is the right thing to be doing
                circs.append(Circle(xc*bin_size_cm , yc*bin_size_cm , r*bin_size_cm ))

        #=============

        if get_area:
            field_area_median = np.median(field_areas[1:])
            field_area_mean = np.mean(field_areas[1:])
            field_area_geomean = gmean(field_areas[1:])

        #=============
        
        return InfoFieldProps(tri=tri,
                          tri_used=tri_used,
                          tri_scale=tri_scale,
                          circs=circs,
                          field_area_median=field_area_median,
                          field_area_mean=field_area_mean,
                          field_area_geomean=field_area_geomean,
                          boundary_labels=boundary_labels,
                          peak_xy=peak_xy)
        
            
    #@profile
    def get_spa_ac_windowed(self, t, c, win_size_sec=30, down_samp_rate=10, bin_size_cm=2.5,
                           smoothing_bins=5, Pthresh=100, pos_mask=None, as_1d=False,
                           win_size_cm=None, return_extra=False, blen=30000):
        """
        Produces a spatial autocorrelogram, by building up a dwell and spike map
        using relative positions, with a window centred on each spike.
        
        ``pos_mask``, if not ``None``, should be a boolean np.array the same length
        as ``self.xy``..only sectiosn of the trial where ``pos_mask`` is ``True``
        will be used, i.e. spieks outside these regions will be ignored, and 
        dwell outside these regions will no be counted.
        Note however, that we are not paritioning the trial into contiguous
        sections of true in the mask, we are just enforcing that no data be
        used form the masked regions...i.e. we could have spikes separated by
        5 seconds, say, where for the middle 2 of those 5 seconds the mask is False, but
        at either end of the 5 secodns the mask is True.
        
        ``as_1d`` - if True the spike counts and dwell counts will be collapsed
        into 1-dimensional distances before binning...otherwise, i.e. if False,
        the spike counts and dwell counts will be binned as 2d offsets from the
        base spike.
        
        You specify the window width either in terms of seconds or path length.
        In both cases the window starts at the spike and projects forwards and
        backwards by the specified amount. ``win_size_sec`` can be the string 
        ``'trial'`` if you want to use the full trial duration.
        Note the pos downsampling isn't done symmetrically - we start at the early 
        end of the window and step forward in the chosen step size. This means
        that we may or may not include the central location itself in the data
        and we may or may not sample the same number of points from forward and
        back.

        You can use large windows if you want.  For large windows we end up 
        batching the historgram computation. The ``blen`` argument controls
        the size of batches, you can play with it a bit to see what is fastest
        for you. 
        
        TODO: this has not been tested properly...I'm not really sure how you'd do that
            except by comparing against someone else's values for the same data.
                
        This function is quite complciated...it started off complicated, and then
        I added a bunch of variations eahc of which required a little if-statement
        here or tehre...I didn't want to make lots of similar but slightly different functions.
        I hope it's still clear enough what is going on here.        
        
        Note the original version of the Matlab code written by NB contained an
        invalid assumption about symmetry of the dwell histogram. The dwell histogram
        is *not* supposed to be symmetric, so the final SAC is also not symmetric.
        (The spikes histogram is symmetric though because for a given pair of 
        spikes the dispalcement between the two points in space is independant
        of which of the two spieks you consider first..you just need to take
        account of the sign.)
        """
                    
        spk_pos_ind = self.tet_times(t, c, as_type='p')
        xy = self.xy/float(bin_size_cm) # Note divding by bin_size_cm here is arithmetically identical to dividing after calculating 1d or 2d displacement
        w, h = self.w/float(bin_size_cm), self.h /float(bin_size_cm)        
        posSampRate = self.pos_samp_rate

        # The down_samp_rate allows us to do a fraction of the work.
        # if the down_samp_rate doesnt divide nicely into the true rate we adjust it.
        downSampStep = int(np.ceil(posSampRate/down_samp_rate))
        down_samp_rate = posSampRate/downSampStep 

        if pos_mask is not None:
            #limit ourselves to only the spikes from where pos_mask=True
            spk_pos_ind = spk_pos_ind[pos_mask[spk_pos_ind]]
            xy[:, ~pos_mask] = np.nan # and record that some pos are bad
        nSpks = len(spk_pos_ind)
        
        # Work out the limits of the histograms and make a binning function that accepts an (nx2) np.array 
        # and calcualte either 1d or 2d binning indices, returning a tuple (inds0, inds1) or just (inds1, ).
        # (note we've alrady transformed xy to be in bin units so no need to divide here, just round to int)
        if as_1d is False:
            mx, my = np.ceil(w), np.ceil(h)        
            AsBinInds = lambda x: tuple((x + np.array([mx+0.5, my+0.5])).astype(int).T)
            hshape = np.array([mx*2+1, my*2+1], dtype=int)
            UseSymmetryOnSpikes = lambda spks: spks + spks[::-1, ::-1]
        else: 
            AsBinInds = lambda x: ((np.hypot(x[:, 0], x[:, 1]) +0.5).astype(int) , )
            mx = np.ceil(np.hypot(w, h))
            hshape = np.array([mx*2 +1], dtype=int)
            UseSymmetryOnSpikes = lambda spks: spks*2
            
        # cache access to xy for spikes..not that important, but do it anyway
        xy_spk_pos_ind = xy[:, spk_pos_ind]        

        if win_size_sec is not None and win_size_cm is not None:
            raise Exception("window width can only be specified in terms of time or distance, not both.")
        win_mode = 'sec' if win_size_sec is not None else 'cm'
        if isinstance(win_size_sec, str) and win_size_sec.lower() == 'trial':
            win_size_sec = self.duration

        """ Compute the displacement histogram for each *pos* in each window."""
        # We do it in batches, collecing the values into a dwell histogram
        
        # Work out the start and end pos sample inds for the window that follows each spike
        if win_mode =='sec' is not None:  # FIXME: what does this even mean? 
            dwell_start_ind = spk_pos_ind - int(posSampRate * win_size_sec) 
            dwell_end_ind = spk_pos_ind + 1 + int(posSampRate * win_size_sec)
        else: # win_mode == 'cm'
            cum_path_length = self.pathLenCum # Note that since we are getting this from the raw xy, we are back in true cms, not bin units
            dwell_start_ind = np.searchsorted(cum_path_length, cum_path_length[spk_pos_ind]-win_size_cm)
            dwell_end_ind = np.searchsorted(cum_path_length, cum_path_length[spk_pos_ind]+win_size_cm)
        dwell_start_ind.clip(0, self.n_pos-1, out=dwell_start_ind)
        dwell_end_ind.clip(0, self.n_pos-1, out=dwell_end_ind)
        
        n_dwell_in_window = np.ceil((dwell_end_ind-dwell_start_ind)/float(downSampStep)).astype(int)
        dwell = np.zeros(hshape, dtype=int)        
        for blen, out_slices, start_inds, end_inds, inds in \
            batched(n_dwell_in_window, dwell_start_ind, dwell_end_ind, np.arange(nSpks), blen=blen):
                
            # collect a list of dx, dy
            dxy = np.empty([blen, 2], dtype=xy.dtype)                
            for slc, start, end, ii in zip(out_slices, start_inds, end_inds, inds):
                dxy[slc] = (xy[:, start:end:downSampStep] - xy_spk_pos_ind[:, [ii]]).T
                
            # we only have to worry about pos_mask for dwell because the spike 
            # list was filtered right at the start..
            if pos_mask is not None:
                dxy = dxy[~np.isnan(dxy)]

            # Get bin index and bin in 1d/2d as requested
            dbin = AsBinInds(dxy)
            dwell += aggregate(dbin, 1, size=hshape)
            

        """ Compute the displacement histogram for each *spike* in each window."""
        # again, we do it in batches, this time producing a spike histogram
        
        spk_start_inds = np.arange(len(spk_pos_ind)) + 1 # here we only go forward, and use symmetry at the end
        spk_end_inds = np.searchsorted(spk_pos_ind, dwell_end_ind)
        n_spikes_in_win = spk_end_inds - spk_start_inds
        spike = np.zeros(hshape, dtype=int)
        n_spikes_in_win[n_spikes_in_win==-1] = 0 # could end up as -1 because spk_start_inds starts at "+1", which may not actually lie in window
        for _, n_in_win, start_inds, inds in \
                batched(n_spikes_in_win, spk_start_inds, np.arange(nSpks), 
                        blen=blen, as_slices=False):
                    
            # get a list of dx, dy.
            a_ind = np.repeat(np.arange(len(n_in_win)), (n_in_win)) + inds[0]
            b_ind = a_ind + 1 + npg.multi_arange(n_in_win)
            dxy = xy_spk_pos_ind[:, b_ind] - xy_spk_pos_ind[:, a_ind]
        
            # Get bin index and bin in 1d/2d as requested
            dbin = AsBinInds(dxy.T)
            spike += aggregate(dbin, 1, size=hshape)
            
        spike = UseSymmetryOnSpikes(spike) # we only binned the spikes going forward

        # Smooth both dwell and spike, do division and mask nodwell
        nodwell = dwell <= Pthresh
        if smoothing_bins is not None and smoothing_bins > 1: 
            kern = np.ones([smoothing_bins])/smoothing_bins if as_1d else np.ones([smoothing_bins, smoothing_bins])/smoothing_bins**2 
            dwell = sp.ndimage.convolve(dwell.astype(np.single), kern, mode='nearest')     
            spike = sp.ndimage.convolve(spike.astype(np.single), kern, mode='nearest')     
        else:
            dwell = dwell.astype(np.single)
            spike = spike.astype(np.single)
        ac = np.ma.array(spike/dwell, mask=nodwell)
        
        if return_extra:
            return ac, bin_size_cm
        else:
            return ac            
    
    def get_rate_ll(self, t, c, return_extra=False, spa_smoothing=1.5,
                  spa_smoothing_type='gaussian', t_smoothing=0.25):
        """
        It uses ``_getSmoothedRate`` to give each moment in time an instantaneuous
        firing rate value.  It then constructs a ratemap (it's a slightly unsual
        ratemap in that it uses this smoothed rate rather than raw spikes, but
        it's basically just a smooth-looking ratemap)...from this ratemap it
        can lookup the np.mean rate at that location.
        
        It then compares the np.mean spatial rate and instantaneous rate by
        assuming the instantaneous rate is being Poission-sampled, from a distribution
        with the stated np.mean.
        
        The probability is caclualted, and the sum of logs is taken to get
        an overal probability for the trial.
        
        Highly negative values correspond to "less likely"/"less stable"/"messier"
        firing...less negative values correspond to "more likely"/"more stable"/"neater"
        firing, roughly speaking.  The longer the trial the more negative the result,
        i.e. it is not normalsied to per unit time.
        
        Note that the smoothing results in significant interdpendencies, which
        means it is definitely not fair to claim that the total values really
        are probabilities.  There's also wierd issues with using the same data
        for estimatig np.mean rate and for computing the probabilty of obersving that
        rate...but here's the code....
        """
        raise DontTrustThisAnalysisException
        
        r_inst = self._getSmoothedRate(t, c, winParam=t_smoothing)
        
        r_mean = self._getRateAtXY(mode='pos',
                                    spk_pos_inds=np.arange(self.n_pos),
                                    spkWeights=r_inst,
                                    smoothing=spa_smoothing,
                                    smoothing_type=spa_smoothing_type)
        
        p = r_mean**r_inst * np.exp(-r_mean) / sp.misc.factorial(r_inst)
        LL = np.sum(np.log(p))                    
        if return_extra:
            return InfoRateLL(LL_total=LL, pos_likelihood=p)
        else:
            return LL
            
    @append_docstring(get_gc_fields)
    def dist_to_peak(self, t, c, **kwargs):
        """
        Based on Hardcastle et al. 2015, however defaults to using a different
        method for isolating fields. See `get_gc_fields(mode=1)`.
        """
                              
        fields_info = self.get_gc_fields(t, c, return_extra=True, **kwargs)                      
        labels, rm = fields_info.labels, fields_info.rm
        
        rm_original = rm.copy()
        rm[labels==0] = 0
        
        # Calculate center of mass for each field..I think you can do with with sp actually
        n_fields = np.max(labels)
        if n_fields is np.ma.masked:
            raise ValueError("No fields found")
            
        x_sum = aggregate(labels.ravel(), 
                          (rm * np.arange(rm.shape[1])[_ax, :]).ravel(),
                          size=n_fields+1)
        y_sum = aggregate(labels.ravel(),
                          (rm * np.arange(rm.shape[0])[:, _ax]).ravel(),
                          size=n_fields+1)
        r_sum = aggregate(labels.ravel(), rm.ravel(), size=n_fields+1)
        x_sum, y_sum, r_sum = x_sum[1:], y_sum[1:], r_sum[1:]
        x = (x_sum/r_sum + 0.5) * fields_info.bin_size_cm
        y = (y_sum/r_sum + 0.5) * fields_info.bin_size_cm
        
        spk_pos_idx = self.tet_times(t, c, as_type='p')
        xy_spike = self.xy[:, spk_pos_idx]
        
        # Calculate radius of fields. The definition appears to be omitted 
        # in the paper, we opt to calcualte the equivalent radius of a 
        # circle with the same area as the field.
        field_radius = np.sqrt(fields_info.field_area/np.pi)
        mean_radius = np.mean(field_radius)
        
        dist = np.sqrt(np.min(\
                (xy_spike[0][:, _ax] - x[_ax, :])**2 + \
                (xy_spike[1][:, _ax] - y[_ax, :])**2 ,
                axis = 1))
        dist /= mean_radius
        return rm_original, dist, y, x
     
        
        
def _get_rotational_corrs(ac, anglesDeg):
    """
    It takes a masked matrix ``ac``, and rotates it by each of the angles given in 
    the list of ``anglesDeg``, rotation uses linear interpolation.

    For each of the rotations it then computes the pearson correlation over all
    the pairs of pixels that are valid in both the source and rotated np.array.
    
    The results are returned in a dict with the angles as keys and the correlations as values.
    """
    if np.ma.is_masked(ac):
        ac = ac.filled(fill_value=np.nan)
    origIsNan = np.isnan(ac)
    
    corrVals = {} 
    for angle in anglesDeg:
        rotatedA = sp.ndimage.rotate(ac, angle=angle, cval=np.nan, reshape=False, order=1)
        
        # we will ignore elements where orignial or rotated has nans
        allNans = origIsNan | np.isnan(rotatedA)
        
        # get the correlation between the original and rotated images and assign
        corrVals[angle], _ = sp.stats.pearsonr(ac[~allNans], rotatedA[~allNans])
        
    return corrVals


def _coherence(x, nodwell, nodwell_is_zeroed=False):
    """
    As described in Muller & Kubie 1989...hopefully.
    """    
    if nodwell_is_zeroed is not True:
        x[nodwell] = 0        
    
    n_valid_neighbors = _neighbor_sum(~nodwell)
    valid_neighbor_mean = _neighbor_sum(x) / n_valid_neighbors 
        
    rho = np.corrcoef(x[~nodwell].ravel(), valid_neighbor_mean[~nodwell].ravel())[0, 1]
    
    return np.arctan(rho) # this is the z-transform
    
    
def _spa_ac_from_rm(x, nodwell, tol=1e-10, nodwell_is_zeroed=False, return_n=False, normalise_n=False):
    """
    For a np.single 2d np.array, ``x``, or a stack of 2d arrays, it calculates the autocorrelation
    at all offsets.  You must provide a nodwell 2d np.array which is True where the
    values in ``x`` are invalid, if the invalid values in x have been set to 0
    then you can indicate that with ``nodwell_is_zeroed=True`` here, but you must
    still provide the nodwell np.array.
    
    FFT and IFFT seem to be fairly slow.  It may be worth padding to improve factorisation.
    
    You can get the overlap counts in addition to the autocor if you set
    ``return_n`` to be ``True``.  If you want counts normalised to the maximum, set ``normalise_n`` 
    to also be ``True`` (this is useful if you are going to treat n as weights or alpha values).
    When ``return_n`` is ``True`` the ac is not a ``MaskedArray``, it is just a normal np.array with nans.
    
    See http://en.wikipedia.org/wiki/Convolution_theorem    
    """
    # We need to zero the no-dwell elements so that they dont contribute    
    if nodwell_is_zeroed is not True:
        x[nodwell] = 0
         # nodwell must be exactly 2d, but x can be 2d or 3d.  Numpy knows what we np.mean for 3d x. TODO: check this.
                
    # For simplicity we force x and nodwell to have a third dimension even 
    # if we're really only dealing with 2d data
    m, n = x.shape[0:2] 
    nodwell = np.reshape(nodwell, [m, n, 1])
    x = np.reshape(x, [m, n, -1])
          
    # [Step 1] Obtain FFTs of x, the sum of squares and bins visited
    Fx = fft(fft(x, 2*m-1, axis=0), 2*n-1, axis=1)
    Fsum_of_squares_x = fft(fft(x*x, 2*m-1, axis=0), 2*n-1, axis=1)
    Fn = fft(fft((~nodwell).astype(int), 2*m-1, axis=0), 2*n-1, axis=1)
    
    # [Step 2] Multiply the relevant transforms and invert to obtain the
    # equivalent convolutions
    rawCorr = fftshift(np.real(ifft(ifft(Fx * np.conj(Fx), axis=1), axis=0)), axes=[0, 1])
    sums_x = fftshift(np.real(ifft(ifft(np.conj(Fx) * Fn, axis=1), axis=0)), axes=[0, 1])
    sum_of_squares_x = fftshift(np.real(ifft(ifft(Fn * np.conj(Fsum_of_squares_x), axis=1), axis=0)), axes=[0, 1])
    n = fftshift(np.real(ifft(ifft(Fn * np.conj(Fn), axis=1), axis=0)), axes=[0, 1])

    
    # [Step 3] Account for rounding errors.
    rawCorr[abs(rawCorr) < tol] = 0
    sums_x[abs(sums_x) < tol] = 0
    sum_of_squares_x[abs(sum_of_squares_x) < tol] = 0
    n = np.ma.array(n.round(), mask=(n<=1))
    
    # [Step 4] Compute correlation matrix
    map_std = np.sqrt((sum_of_squares_x * n) - sums_x*sums_x)
    map_covar = (rawCorr * n) - sums_x * sums_x[::-1, ::-1, :]
    
    
    ret = np.squeeze(map_covar / map_std / map_std[::-1, ::-1, :]).clip(-1, 1)    

    if not return_n:
        return ret
    else:
        if normalise_n:    
            n = n.squeeze()
            center = (int(n.shape[0]/2), int(n.shape[1]/2))
            n /= n[center];      
        # TODO: it was a bit pointless using masked arrays above, only to use nans here
        return (ret.filled(np.nan), n.filled(0))
    
        
def _get_orientation(peaks_coord_central, orientation_mode='first'):
    if peaks_coord_central.shape[0] == 1:
        return np.nan
    else:
        theta = np.arctan2(peaks_coord_central[:, 0], -peaks_coord_central[:, 1])  * (180/ np.pi)
        if orientation_mode == 'first':
            return np.sort(theta[theta>0])[0]
        elif orientation_mode == 'mean':
            return _circ_mean_60degrees(theta)
        else:
            raise Exception("what?")

def _im_regional_max(im, use_centroid=False, min_peak_height=0):
    """
    takes an image and returns an image of the same size, with np.zeros everywhere apart
    from a a set of np.single pixels, which are uniquely labeled, at the maxima.
    
    When ``use_centroid`` is ``False``, where the maxima are not np.single pixels, 
    they are abitrarilay reduced to np.single pixels. (the most bottom right point 
    in the peak will win)
    
    Alternatiebly, when ``use_centroid``is ``True`` the maxima are reduced to np.single
    pixels by taking the centroid and rounding.
    """

    if np.ma.is_masked(im):
        im = im.filled(fill_value=0)
        
    # simple imregional max
    peaks_mask =  (im>min_peak_height) & (sp.ndimage.grey_dilation(im, 
                                                            size=(3, 3)) == im)
    peaks_label = skimage.measure.label(peaks_mask, 8)
    
    if use_centroid is True:
        regProps = skimage.measure.regionprops(peaks_label)
        peaksXYCoord = np.array([reg['Centroid'] for reg in regProps]) #note that this works like a LazyDict
        peaksXYCoord = round(peaksXYCoord).astype(int)
    else:
        # if each peak consists of more than 1 pixel, but not more than 2 or 3 then we can do this..
        extendedPeaksXYCoord = peaks_mask.nonzero()
        extendedPeaksLabels = peaks_label[peaks_mask]
        if len(extendedPeaksLabels):
            peaksXYCoord = np.empty((2, np.amax(extendedPeaksLabels)), int) 
            peaksXYCoord[:, extendedPeaksLabels-1] = np.array(extendedPeaksXYCoord) # forgets all but the last item for each label 
        else:
            peaksXYCoord = np.empty((2, 0), dtype=int)
            
    # we recreate the peaks_label using the peaksXYCoord...
    peaks_label = np.zeros(peaks_mask.shape, np.int8)
    peaks_label[[peaksXYCoord[0, :], peaksXYCoord[1, :]]] = np.arange(1, 1+peaksXYCoord.shape[1])
    
    return peaks_label, peaksXYCoord
        
        
def _find_peak_extent_multi(ac, peaks, thresh_frac=0.5):
    """ 
    This is different to _find_peak_extent in two ways: firstly, here we 
    operate on all peaks in one go; secondly, we use watersheding to define
    the area around the peak which gets thresholded, rather than thresholding
    the whole matrix and then labeleing taking the contiguous region around the peak.
    
    ac is scaled and cast to uint8. (I think it could be uint16, but who cares?)
    
    You can supply either a labeled np.array the same size as ac for the peaks or
    a list of coordinates. Unless ac is nx2 it should be obvious what you np.mean.
    ..and even then you could still disambiguate if neccessary.
    """    
    
    if ac.dtype is not np.dtype(np.uint8): # TODO: this is not the correct way of testing!!!
        Auint8 = (ac-np.amin(ac))
        m = float(np.amax(Auint8))
        if m == 0:
            return np.zeros(ac.shape, dtype=np.uint8)
        Auint8 *= 255./m
        Auint8 = Auint8.astype(np.uint8)
    else:
        Auint8 = ac
        
    if peaks.shape == ac.shape: #Note we don't actually deal with the possibility that ac is nx2
        labeledPeaks = peaks
    else:
        labeledPeaks = np.zeros_like(ac, int)
        labeledPeaks[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks)+1)
    
    peaks_mask = labeledPeaks.astype(bool)
    labeledSheds = skimage.segmentation.watershed(255-Auint8, labeledPeaks)
    
    # Define a threshold for each watersed, using some fraction of the peak rate
    shedAVal = np.zeros(np.max(labeledPeaks)+1)
    shedAVal[labeledPeaks[peaks_mask]] = ac[peaks_mask]
    shedAVal *= thresh_frac
    
    # Apply the thresholds, to crop each of the sheds down to being a field
    labeledSheds[ac<shedAVal[labeledSheds]] = 0 
    return labeledSheds
    
def _find_peak_extent(ac, peakCoord, thresh_frac=0.5, thresh_abs=None):
    '''
    For a matrix ac, and a coordiante in that matrix, peakCoord, it returns
    a mask of the same shape as ac, with  Trues showing the region around 
    the peak with values greater than thresh_frac times the peak's value.
    '''
    if thresh_abs is None:
        thresh = ac[peakCoord[0], peakCoord[1]] * thresh_frac
    else:
        thresh = thresh_abs
    above_thresh_labels = skimage.measure.label(ac > thresh, 8)
    peakIDTmp = above_thresh_labels[peakCoord[0], peakCoord[1]]
    return above_thresh_labels == peakIDTmp 


def _neighbor_sum(ac):
    """
    ac is a 2d np.array.
    Return value is the sum of the 8 neigbours at each position in the np.array.
    """
    
    ret = uniform_filter(ac.astype(np.single), size=3, mode='constant')*9; #note that if ac was small ints then then uniform_filter gives meaningless results
    ret -= ac
    return ret
    
def _boundary_mask(ac):
    """
    Returns a mask that is True wherever one of the 4-neighbors of a pixel is not 
    equal to the pixel.
    """
    isBoundary = np.zeros(ac.shape, dtype=bool)
    isBoundary[1:, :] |= ac[1:, :] != ac[:-1, :]
    isBoundary[:-1, :] |= ac[1:, :] != ac[:-1, :]
    isBoundary[:, 1:] |= ac[:, 1:] != ac[:, :-1]
    isBoundary[:, :-1] |= ac[:, 1:] != ac[:, :-1]
    return isBoundary
    

def sft2_low(x, maxFreq, totalLen, duplicate=True, return_extra=False):
    """ This was first written by daniel in Matlab. This Python version seems 
        to give roughly the right answers.
        TODO: test properly.
    """
    N_1, N_2 = x.shape;
    
    if not isinstance(totalLen, (list, tuple)):
        totalLen = [totalLen]*2
    if not isinstance(maxFreq, (list, tuple)):
        maxFreq = [maxFreq]*2
            
    k_list_1 = np.matrix(np.arange(0, maxFreq[0]*totalLen[0]/N_1 +1, 
                                   dtype=np.double), copy=False)
    k_list_2 = np.matrix(np.arange(0, maxFreq[1]*totalLen[1]/N_2 +1, 
                                   dtype=np.double), copy=False)
    
    f = np.exp(-2*np.pi*1j/totalLen[1] * k_list_2.transpose() * np.matrix(
                                np.arange(0, N_2), copy=False)) * x.transpose() # dim=2
    f = np.vstack((f, f[:0:-1, :].np.conj()))
    
    f = f * np.exp(-2*np.pi*1j/totalLen[0] * np.matrix(
                            np.arange(0, N_1), copy=False).transpose() * k_list_1) # dim=1
    f = f.transpose()
    
    if duplicate:
        f = np.vstack((f , np.hstack((f[:0:-1, 0].np.conj(), f[:0:-1, :0:-1].np.conj()))))
    
    f = np.asarray(f) # dont want to worry anyone else about np matrices
    
    if return_extra is False:
        return f
    else:
        k_list_1, k_list_2 = np.asarray(k_list_1).ravel(), np.asarray(k_list_2).ravel()
        freqs2 = np.hstack((k_list_2, -k_list_2[:0:-1])) * N_2/totalLen[1];
        if duplicate:
            freqs1 = np.hstack((k_list_1, -k_list_1[:0:-1])) * N_1/totalLen[0];
        else:
            freqs1 = k_list_1 * N_1/totalLen[0];
        return f, freqs1, freqs2


def _circ_mean_60degrees(x, axis=None):
    """
    input is a vector of values in degrees on the interval [0, 60), the output
    is the circular average of those values, where we imagine that [0, 60) is 
    the full circle, i.e. 0=60.
    
    If x has more than two dimensions, ``axis`` controls the dimension of
    the np.mean taking.
    """
    x = np.asanyarray(x)
    return np.angle(np.sum(np.exp(1j*x/180.0*np.pi*6), axis=axis), deg=True)/6
    
""" This section of code was pulled out of the get_gc_measures function """
# attempt to fit an ellipse to the closest peaks 
#        try:
#            a = self.__fit_ellipse__(closest_peaks_coord[1:, 0], closest_peaks_coord[1:, 1])
#            im_centre = self.__ellipse_center__(a)
#            ellipse_axes = self.__ellipse_axis_length__(a)
#            ellipse_angle = self.__ellipse_angle_of_rotation__(a)
##            ang =  ang + pi
#            ellipseXY = self.__getellipseXY__(ellipse_axes[0], ellipse_axes[1], ellipse_angle, im_centre)
#            # get the minimum containing circle based on the minor axis of the ellipse
#            circleXY = self.__getcircleXY__(im_centre, min(ellipse_axes))
#        except:
#            im_centre = central_point
#            ellipse_angle = np.nan
#            ellipse_axes = (np.nan, np.nan)
#            ellipseXY = central_point
#            circleXY = central_point
 
""" Here are a few more functions from Robin's code that don't concern me right now"""       


    
#    def deformSAC(self, ac, circleXY, ellipseXY):
#        tform = skimage.transform.AffineTransform()
#        tform.estimate(ellipseXY, circleXY)
#        '''
#        the transformation algorithms used here crop values < 0 to 0. Need to 
#        rescale the SAC values before doing the deformation and then rescale 
#        again so the values assume the same range as in the unadulterated SAC
#        '''
#        SACmin = min(ac.flatten())#should be 1
#        SACmax = max(ac.flatten())
#        AA = ac + 1
#        deformedSAC = skimage.transform.warp(AA / max(AA.flatten()), inverse_map=tform.inverse, cval=0)
#        return skimage.exposure.rescale_intensity(deformedSAC, out_range=(SACmin, SACmax))
#           
#    
#
#    def crossCorr2D(self, ac, B, A_nodwell, B_nodwell, tol=1e-10):
#        [ma, na, oa] = ac.shape
#        [mb, nb, ob] = B.shape
#        ac = np.reshape(ac, (ma * na, oa))
#        B = np.reshape(B, (mb * nb, ob))
#        ac[A_nodwell.ravel(), :] = 0
#        B[B_nodwell.ravel(), :] = 0
#        ac = np.reshape(ac, (ma, na, oa))
#        B = np.reshape(B, (mb, nb, ob))
#        
#        # [Step 1] Obtain FFTs of x, the sum of squares and bins visited
#        Fa = fft(fft(ac, 2*mb-1, axis=0), 2*nb-1, axis=1)
#        Fsum_of_squares_a = fft(fft(power(ac, 2), 2*mb-1, axis=0), 2*nb-1, axis=1)
#        Fn_a = fft(fft(invert(A_nodwell).astype(int), 2*mb-1, axis=0), 2*nb-1, axis=1)
#       
#        Fb = fft(fft(B, 2*ma-1, axis=0), 2*na-1, axis=1)
#        Fsum_of_squares_b = fft(fft(power(B, 2), 2*ma-1, axis=0), 2*na-1, axis=1)
#        Fn_b = fft(fft(invert(B_nodwell).astype(int), 2*ma-1, axis=0), 2*na-1, axis=1)
#
#        # [Step 2] Multiply the relevant transforms and invert to obtain the
#        # equivalent convolutions
#        rawCorr = fftshift(np.real(ifft(ifft(Fa * np.conj(Fb), axis=1), axis=0)))
#        sums_a = fftshift(np.real(ifft(ifft(np.conj(Fa) * Fn_b, axis=1), axis=0)))
#        sums_b = fftshift(np.real(ifft(ifft(np.conj(Fn_a) * Fb, axis=1), axis=0)))
#        sum_of_squares_a = fftshift(np.real(ifft(ifft(Fsum_of_squares_a * np.conj(Fn_b), axis=1), axis=0)))
#        sum_of_squares_b = fftshift(np.real(ifft(ifft(Fn_a * np.conj(Fsum_of_squares_b), axis=1), axis=0)))
#        n = fftshift(np.real(ifft(ifft(Fn_a * np.conj(Fn_b), axis=1), axis=0)))
#        
#        # [Step 3] Account for rounding errors.
#        rawCorr[abs(rawCorr) < tol] = 0
#        sums_a[abs(sums_a) < tol] = 0
#        sums_b[abs(sums_b) < tol] = 0
#        sum_of_squares_a[abs(sum_of_squares_a) < tol] = 0
#        sum_of_squares_b[abs(sum_of_squares_b) < tol] = 0
#        n = round(n)
#        n[n<=1] = np.nan
#        
#        # [Step 4] Compute correlation matrix
#        map_std_a = sqrt((sum_of_squares_a * n) - sums_a**2)
#        map_std_b = sqrt((sum_of_squares_b * n) - sums_b**2)
#        map_covar = (rawCorr * n) - sums_a * sums_b[::-1, ::-1, :]
#        
#        return squeeze(map_covar / map_std_a / map_std_b[::-1, ::-1, :])
