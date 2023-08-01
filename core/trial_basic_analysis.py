# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp

from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import firwin as sp_firwin
from utils.custom_functools import append_docstring
from utils.custom_np import argsort_inverse, reindex_masked, ignore_divide_by_zero
from numpy_groupies import aggregate_np as aggregate
import numpy_groupies as npg
from noncore.adaptive_ratemap import get_ratemap_adaptive


class TrialBasicAnalysis(object):
    """This is a mixin for use with trial.TT"""

    """ pure-pos stuff """

    def xy_bin_inds(self, bin_size_cm=2.0, xy=None, w=None, h=None):
        """ returns (idx, w, h), where idx is shape=[2,n_pos] and w and h 
            are the numbers of bins in each dimension. """
        xy = self.xy if xy is None else xy
        w = self.w if w is None else w
        h = self.h if h is None else h
        return (xy / bin_size_cm).astype(int), \
               int(np.ceil(w / float(bin_size_cm))), \
               int(np.ceil(h / float(bin_size_cm)))

    def speed_bin_inds(self, min_cm_per_s=5, bin_size_cm_per_s=4,
                       max_cm_per_s=45, return_extra=False):
        """ returns (idx, 0, hi_idx), where idx is shape=n_pos, and hi_idx is the
        maxiumum bin number.  Spikes below min_cm_per_s go into bin 0, spikes above
        max_cm_per_s go into bin hi_idx."""

        n_valid_bins = np.ceil(float(max_cm_per_s - min_cm_per_s)
                               / bin_size_cm_per_s)
        speed_bin_ind = (self.speed - min_cm_per_s) / bin_size_cm_per_s
        speed_bin_ind = np.clip(np.floor(speed_bin_ind).astype(int) + 1,
                                0, int(n_valid_bins) + 1)

        if return_extra:
            bin_lefts = np.r_[0, min_cm_per_s + np.arange(n_valid_bins + 1)
                              * bin_size_cm_per_s]
            bin_widths = np.r_[min_cm_per_s, [bin_size_cm_per_s] * int(n_valid_bins),
                               bin_size_cm_per_s * 1.5]  # note last bin width is arbitrary
            return speed_bin_ind, (0, int(n_valid_bins) + 1), bin_lefts, bin_widths
        else:
            return speed_bin_ind, (0, int(n_valid_bins) + 1)

    def dir_bin_inds(self, bin_size_deg=5.0, force_use_disp=False):
        """ returns idx, bin_left_edge. """
        if 360 % bin_size_deg != 0:
            raise ValueError("360 must divide by binSizeDeg without remainder.")
        n_bins = 360 / bin_size_deg
        dir_ = self.dir_disp if force_use_disp else self.dir
        direc = dir_ + np.pi
        bin_size_rad = bin_size_deg / 180.0 * np.pi
        bin_idx = np.floor(direc / bin_size_rad).astype(int)
        bin_idx[bin_idx == n_bins] = 0  # wrap
        return bin_idx, np.arange(0, n_bins) * bin_size_deg

    @property
    def path_len_cum(self):
        """ len=n_pos, cumulative length in cm """
        if not self._cache_has('pos', '_cache_path_len_cum'):
            self._cache_path_len_cum = np.cumsum(np.hypot(
                np.ediff1d(self.xy[0], to_begin=0),
                np.ediff1d(self.xy[1], to_begin=0)))
        return self._cache_path_len_cum

    """ pure theta/eeg stuff """

    @property
    def theta_freq_band(self):
        """ returns (low_freq, high_freq) in Hz. """
        if not hasattr(self, '_theta_freq_band'):
            self._theta_freq_band = (6, 12)
        return self._theta_freq_band

    @theta_freq_band.setter
    def theta_freq_band(self, v):
        """ lets the user set (low_freq, high_freq) in Hz. """
        self._clear_cache(drop=('theta',))
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError("Expected (low_freq, high_freq).")
        self._theta_freq_band = tuple(v)

    def filtered_eeg(self, bad_as='nan'):
        """ returns the eeg band-pass filtered for the theta_freq_band
        using a 1second-tap bandpass blackman filter."""
        if not self._cache_has('eeg theta', '_cache_filtered_eeg'):
            nyquist = self.eeg_samp_rate / 2
            eeg = self.eeg(bad_as=0)
            eeg_filter = sp_firwin(int(self.eeg_samp_rate + 1),
                                   tuple(f / nyquist for f in self.theta_freq_band),
                                   window='black', pass_zero=False)
            filtered_eeg = sp.signal.filtfilt(eeg_filter, [1], eeg)
            filtered_eeg.setflags(write=False)
            self._cache_filtered_eeg = filtered_eeg

        if bad_as is None or self.eeg_is_bad is None:
            return self._cache_filtered_eeg
        else:
            return np.where(self.eeg_is_bad, np.nan if bad_as == 'nan' else bad_as,
                            self._cache_filtered_eeg)

    def filtered_egf(self, bad_as='nan'):
        """ returns the eeg band-pass filtered for the theta_freq_band
        using a 1second-tap bandpass blackman filter."""
        if not self._cache_has('eeg theta', '_cache_filtered_eeg'):
            nyquist = self.egf_samp_rate / 2
            eeg = self.egf(bad_as=0)
            eeg_filter = sp_firwin(int(self.egf_samp_rate + 1),
                                   tuple(f / nyquist for f in self.theta_freq_band),
                                   window='black', pass_zero=False)
            filtered_eeg = sp.signal.filtfilt(eeg_filter, [1], eeg)
            filtered_eeg.setflags(write=False)
            self._cache_filtered_eeg = filtered_eeg

        if bad_as is None or self.egf_is_bad is None:
            return self._cache_filtered_eeg
        else:
            return np.where(self.egf_is_bad, np.nan if bad_as == 'nan' else bad_as,
                            self._cache_filtered_eeg)

    def _analytic_eeg(self, bad_as='nan'):
        """ returns the result of hilbert transform on filtered eeg. """
        if not self._cache_has('eeg theta', '_cache_analytic_eeg'):
            eeg = self.filtered_eeg(bad_as=0)
            # If we don't pad, then we may end up with a length that has very large prime factors
            # that would be very bad indeed when it comes to doing the fft (within the hilbert)    
            # As it happens, the data len should already be a multiple of the sample rate,
            # but if it's not we enforce it now...actually, you might want to add a x10 or
            # something for a small, but significant speed benefit (often/awlays trials get
            # assigned a length that is 1s longer than it should be, which ruins the ease of
            # factorsiing the length).
            padFactor = int(self.eeg_samp_rate)
            padding = (padFactor - len(eeg)) % padFactor
            if padding > 0:
                eeg = np.hstack((eeg, np.zeros(padding)))
            analytic_eeg = sp.signal.hilbert(eeg)
            if padding > 0:
                analytic_eeg = analytic_eeg[:-padding - 1]
            self._cache_analytic_eeg = analytic_eeg
            self._cache_analytic_eeg.setflags(write=False)

        if bad_as is None or self.eeg_is_bad is None:
            return self._cache_analytic_eeg
        else:
            return np.where(self.eeg_is_bad, np.nan if bad_as == 'nan' else bad_as,
                            self._cache_analytic_eeg)

    @property
    def theta_phase(self):
        """ in radians. """
        if not self._cache_has('eeg theta', '_cache_theta_phase'):
            self._cache_theta_phase = np.angle(self._analytic_eeg(None)) + np.pi
            if self.eeg_is_bad is not None:
                self._cache_theta_phase[self.eeg_is_bad] = np.nan
            self._cache_theta_phase.setflags(write=False)
        return self._cache_theta_phase

    @property
    def theta_freq(self):
        """ in Hz. """
        if not self._cache_has('eeg theta', '_cache_theta_freq'):
            phase = self.theta_phase
            if self.eeg_is_bad is not None:
                phase = np.where(self.eeg_is_bad, 0, phase)
            phase = np.unwrap(phase)
            self._cache_theta_freq = np.ediff1d(phase, to_end=[0]) \
                                     * (self.eeg_samp_rate / 2 / np.pi)
            if self.eeg_is_bad is not None:
                self._cache_theta_freq[self.eeg_is_bad] = np.nan
            freq_is_bad = (self._cache_theta_freq < self.theta_freq_band[0]) | \
                          (self._cache_theta_freq > self.theta_freq_band[1])
            self._cache_theta_freq[freq_is_bad] = np.nan
            self._cache_theta_freq.setflags(write=False)
        return self._cache_theta_freq

    @property
    def theta_amp(self):
        """ in volts."""
        if not self._cache_has('eeg theta', '_cache_theta_amp'):
            self._cache_theta_amp = np.abs(self._analytic_eeg())
            if self.eeg_is_bad is not None:
                self._cache_theta_amp[self.eeg_is_bad] = np.nan
            self._cache_theta_amp.setflags(write=False)
        return self._cache_theta_amp

    """ rate-related stuff """

    def get_t_ac(self, t, c, window_width_ms=500., bin_width_ms=2.5,
                 return_extra=False, spike_mask=None):
        """for spikes on tetrode t, cell c, this returns the temporal autocorrelgoram
        using the bin width specified in miliseconds, and maximum delta also 
        specified in miliseconds. """
        times = self.spk_times(t, c)
        if spike_mask is not None:
            times = times[spike_mask]
            mean_rate = None
        else:
            mean_rate = len(times) / self.duration
        window_width_s = window_width_ms / 1000.
        bin_width_s = bin_width_ms / 1000.
        counts = times.searchsorted(times + window_width_s, side='left') \
                 - np.arange(len(times))
        b_idx = np.repeat(np.arange(len(counts)), counts)
        a_idx = npg.multi_arange(counts) + b_idx
        y = times[a_idx] - times[b_idx]
        h = aggregate(np.floor(y / bin_width_s).astype(int), 1,
                      size=int(window_width_s / bin_width_s))
        if return_extra:
            return h, np.arange(len(h) + 1) * bin_width_ms, mean_rate
        else:
            return h

    def get_dir_ratemap(self, t, c, bin_size_deg=5, smoothing_bins=5, pos_mask=None,
                        speed_thresh_cm_per_s=5.0, spk_pos_idx=None,
                        smoothing_type='boxcar', force_use_disp=False):
        """ for spikes on tetrode t, cell c, this returns a directional ratemap, 
        using the specified bin size in degrees and smoothing type and width. 
         
        `speed_thresh_cm_per_s` - whreever speed is less than this value we mask pos.
        
        returns `rm, bin_left_edge` in radians
        """

        # combine pos_mask and speed_thresh requirement
        if speed_thresh_cm_per_s is not None:
            speed_mask = self.speed > speed_thresh_cm_per_s
            if pos_mask is not None:
                pos_mask &= speed_mask
            else:
                pos_mask = speed_mask

        # get spk_pos_idx, although it could be None if we just want dwell
        if t is not None and c is not None and spk_pos_idx is None:
            spk_pos_idx = self.spk_times(t=t, c=c, as_type='p')

        dwell_bin_idx, bin_lower_end = self.dir_bin_inds(bin_size_deg=bin_size_deg,
                                                         force_use_disp=force_use_disp)
        n_bins = len(bin_lower_end)
        spk_bin_idx = dwell_bin_idx[spk_pos_idx] if spk_pos_idx is not None else None

        # apply pos_mask to dwell and spk
        if pos_mask is not None:
            dwell_bin_idx = dwell_bin_idx[pos_mask]
            spk_bin_idx = spk_bin_idx[pos_mask[spk_pos_idx]] if spk_pos_idx is not None else None

        dwell = aggregate(dwell_bin_idx, 1. / self.pos_samp_rate, size=n_bins)
        spk = aggregate(spk_bin_idx, 1, size=n_bins) if spk_pos_idx is not None else None

        if smoothing_bins > 0:
            if smoothing_type == 'boxcar':
                filter_func = sp.ndimage.uniform_filter1d
            elif smoothing_type == 'gaussian':
                filter_func = sp.ndimage.filters.gaussian_filter1d
            else:
                raise Exception("unknown smoothing type: %s" % (smoothing_type))
            dwell = filter_func(dwell, smoothing_bins, mode='wrap')
            spk = filter_func(spk, smoothing_bins, mode='wrap') if spk_pos_idx is not None else None

        if spk is None:
            return dwell, bin_lower_end
        else:
            return spk.astype(float) / dwell.astype(float), bin_lower_end

    def get_spa_ratemap(self, t=None, c=None, bin_size_cm=2.5, smoothing_bins=5,
                        xy=None, w=None, h=None, spk_pos_idx=None, spk_weights=None,
                        pos_mask=None, smoothing_type='boxcar', norm=None,
                        nodwell_mode='ma', spike_shifts_s=None):
        """for spikes on tetrode t, cell c, this returns a spatial ratemap, using
        the specified bin size in cm and smoothing type and width.
        
        `norm` - ``None`` means give rate in Hz, ``pos`` means rate in
         Hz times by pos_samp_rate, i.e. rate per pos sample; ``counts`` means use raw spike
         counts (and ignore dwell time entirely). Alternatively `'max'` to divide
         by peak  rate or `'mean'` to divide by the cell's average firing rate throughout
         the trial.  Or an integer on the interval `(0,100]` to use the given
         percentile in the (valid part of the) ratemap as the normalizer (e.g.
         50 means normalize to median. Or `'rank'` in which case each rate is
         replaced by its percentile rank in the list of valid bins.
                 
        ``spk_weights``, if not None should be the same length as the number of spikes.
        It will be used in the accumulation instead of 1, i.e. each spike can be given
        a specific weighting rather than just being counted.
        
        In addition of the usual ``boxcar``,  ``gaussian``, and ``None``,
        ``smoothing_type`` can be ``adaptive``, in which case the ratemap request
        is passed on to ``get_ratemap_adaptive``, with  ``sigma`` set to 
        ``smoothing_bins`` (try using sigma=200).   
                        
        If ``t`` and ``c`` are not ``None``, it returns a ratemap, with bins in Hz.
        Otherwise it returns a dwell map with bins in seconds. If ``as_count`` is 
        ``True`` and ``t`` and ``c`` are integers, it returns a ratemap with the bins
        giving spike counts per bin, i.e. the division by the dwell map is not done. 
        
        If ``spike_shifts_s`` is not ``None``, it is an array of temporal
        shifts to apply to the spike train, computing a separate ratemap for
        each shift, ie. the result is a stack of ratemaps not a single ratemp.
        Support for this option may have limitations.
        """

        bin_size_cm = float(bin_size_cm)

        # perpare xy, w, h, and spk_pos_idx (if not explicitly provided)
        xy = self.xy if xy is None else xy
        w = self.w if w is None else w
        h = self.h if h is None else h
        if spk_pos_idx is not None:
            pass
        elif t is not None:
            spk_pos_idx = self.spk_times(t=t, c=c, as_type='p')
        elif c is not None:
            if self.recording_type == 'npx':
                spk_pos_idx = self.spk_times(c=c, as_type='p')
            else:
                raise ValueError("if you provide cell number you must also give tet number.")
        # else : only dwell required not true "ratemap"

        if smoothing_type == 'adaptive':
            # RETURN adaptive ratemap using external function...
            if spk_pos_idx is None:
                raise Exception("you must provide spikes for adaptive ratemap.")
            if spk_weights is not None:
                raise NotImplementedError("custom spk_weights not implemented for adaptive smoothing")
            if spike_shifts_s is not None:
                raise NotImplementedError("shifts not implremented for adaptive smoothing.")
            if pos_mask is not None:
                xy = xy[:, pos_mask]
                spk_pos_idx = reindex_masked(pos_mask, spk_pos_idx)

            if self.pos_shape is not None:
                mask_func = self.pos_shape.is_outside
            else:
                mask_func = None
            rm, _ = get_ratemap_adaptive(xy, spk_pos_idx, w, h,
                                         spacing_cm=bin_size_cm,
                                         pos_samp_rate=self.pos_samp_rate,
                                         r_max=15, alpha=smoothing_bins,
                                         mask_func=None,
                                         nodwell_mode=nodwell_mode)
            return rm.T, bin_size_cm
        elif smoothing_type == 'boxcar' and smoothing_bins:
            filter_func = lambda x: sp.ndimage.uniform_filter(x,
                                                              smoothing_bins if np.ndim(x) == 2 else
                                                              (smoothing_bins, smoothing_bins, 0), mode='constant')
        elif smoothing_type == 'gaussian' and smoothing_bins:
            filter_func = lambda x: sp.ndimage.gaussian_filter(x,
                                                               smoothing_bins if np.ndim(x) == 2 else
                                                               (smoothing_bins, smoothing_bins, 0), mode='constant')
        elif smoothing_type is None or not smoothing_bins:
            filter_func = lambda x: x
        else:
            raise ValueError("unknown smoothing type: {}".format(smoothing_type))

        xy_idx, w, h = self.xy_bin_inds(bin_size_cm, xy=xy, w=w, h=h)

        # prepare dwell and/or nodwell
        dwell = aggregate(xy_idx[::-1, pos_mask if pos_mask is not None else slice(None)], 1., size=[h, w])
        nodwell = dwell == 0
        if norm == 'counts':
            pass  # don't care about dwell any more (only nodwell was needed)
        else:
            dwell = filter_func(dwell)
            if norm is None:
                dwell /= self.pos_samp_rate  # now in seconds, otherwise relative norm

        if spk_pos_idx is None:
            rm = dwell
        else:
            # prepare spikes and rate, and normalise rate                
            if spike_shifts_s is None:
                spk = aggregate(xy_idx[::-1, spk_pos_idx], spk_weights or 1.,
                                size=[h, w])
            else:
                # deal with shuffles...
                if spk_weights is not None:
                    raise NotImplementedError("spk_weights not supported for shifted spikes.")
                spike_shifts_pos = (spike_shifts_s * self.pos_samp_rate).astype(
                    int)  # this is not perfect, but good enough
                spike_shifts_pos.shape = 1, -1
                n_shifts = spike_shifts_pos.size
                n_spikes = spk_pos_idx.size
                spk_pos_idx = spk_pos_idx[:, np.newaxis] + spike_shifts_pos
                spk_pos_idx %= int(self.n_pos)
                plane_idx = np.repeat(np.arange(n_shifts)[np.newaxis, :], n_spikes, axis=0)
                spk = aggregate([x.ravel() for x in
                                 tuple(xy_idx[::-1, spk_pos_idx]) + (plane_idx,)],
                                1., size=[h, w, n_shifts])
                dwell = dwell[..., np.newaxis]
                nodwell = np.repeat(nodwell[..., np.newaxis], n_shifts, axis=-1)

            spk = filter_func(spk)
            if norm == 'counts':
                rm = spk
            else:
                with ignore_divide_by_zero():
                    rm = spk / dwell
                if norm is None:
                    pass  # true Hz
                elif norm == "pos":
                    pass  # rate in per-pos-sample units
                elif norm == 'max':
                    rm *= 1. / np.nanmax(rm)
                elif norm == 'mean':
                    rm *= self.duration / len(spk_pos_idx)
                elif norm == 'rank':
                    mask = ~np.isnan(rm)
                    rm[mask] = argsort_inverse(rm[mask]) * (100.0 / np.sum(mask))
                elif 0 < norm <= 100:
                    rm *= 1.0 / np.nanpercentile(rm, norm)  # eg. norm by median
                else:
                    raise ValueError("unknown norm type: {}".format(norm))

        # apply nodwell_mode            
        if nodwell_mode == 'nan':
            rm[nodwell] = np.nan
        elif nodwell_mode == 'ma':
            rm = np.ma.array(rm, mask=nodwell)
        elif np.isscalar(nodwell_mode) and isinstance(nodwell_mode, (float, int)):
            rm[nodwell] = nodwell_mode
        elif isinstance(nodwell_mode, str) and nodwell_mode.startswith('interp'):
            # this is very basic, nearest-neighbor interp, with no averaging for ties.
            nearest_idx0, nearest_idx1 = sp.ndimage.distance_transform_edt(
                nodwell, return_indices=True, return_distances=False)
            rm[nodwell] = rm[nearest_idx0[nodwell], nearest_idx1[nodwell]]
        else:
            raise Exception("unrecognised nodwell mode")

        return rm, bin_size_cm

    def get_t_rate(self, t, c, smoothing_s=0.5, as_type='p', smoothing_type='gaussian',
                   gauss_order=0, as_count=False):
        """For spikes on tetrode t, cell c, this returns the rate in Hz at each time
        point, using pos sampling freq (when `as_type='p'`). If `as_counts` is true,
        the raw counts are returned instead of Hz.
            
        ``smoothing_type`` can be ``gaussian`` in which case ``smoothing`` is the
        standard deviation in seconds...or...``boxcar`` in which case
        ``smoothing`` is the window size in seconds.  When `gaussian`, 
        ``gauss_order`` is passed through to the filter. Or it can be ``None``.    
        """
        if as_type != 'p':
            raise NotImplementedError()

        spk_pos_idx = self.spk_times(t=t, c=c, as_type='p')
        pos_spike_counts = aggregate(spk_pos_idx, 1, size=int(self.n_pos)).astype(np.double)

        if as_count:
            if smoothing_type is not None:
                raise ValueError("when requesting counts, smoothing is not supported.")
            return pos_spike_counts

        rate = pos_spike_counts
        rate *= self.pos_samp_rate  # convert to Hz

        if smoothing_type == 'boxcar':
            return uniform_filter1d(rate, size=smoothing_s * self.pos_samp_rate,
                                    mode='nearest')
        elif smoothing_type == 'gaussian':
            return gaussian_filter1d(rate, sigma=smoothing_s * self.pos_samp_rate,
                                     order=gauss_order, mode='nearest')
        elif smoothing_type is None:
            return rate

    @append_docstring(get_spa_ratemap)
    def get_rate_at_xy(self, t, c, mode='spike', spk_pos_idx=None, norm=None, **kwargs):
        """ reads the rate from the ratemap.  If mode is 'spike' it returns the
            spatial rate at the location of each spike, if 'pos' is returns the 
            spatial rate at the location of each pos sample.  kwargs are passed
            through to get_spa_ratemap."""

        spk_pos_idx = self.spk_times(t, c, as_type='p') if spk_pos_idx is None else spk_pos_idx
        rm, bin_size_cm = self.get_spa_ratemap(spk_pos_idx=spk_pos_idx,
                                               nodwell_mode=np.nan, **kwargs)
        xy_idx, _, _ = self.xy_bin_inds(bin_size_cm)

        if mode == 'pos':
            return rm[xy_idx[1, :], xy_idx[0, :]]
        elif mode == 'spike':
            return rm[xy_idx[1, spk_pos_idx], xy_idx[0, spk_pos_idx]]

    @append_docstring(speed_bin_inds)
    def get_speed_hist(self, return_extra=False, **kwargs):
        """ ``kwargs`` passed through to ``speed_bin_inds``
            TODO: implement speed/rate thing for tc.
        """

        idx, (low_idx, high_idx), bin_lefts, bin_widths = \
            self.speed_bin_inds(return_extra=True, **kwargs)

        h = aggregate(idx, 1, func='sum', size=high_idx + 1)
        if return_extra:
            return h, bin_lefts, bin_widths
        else:
            return h

    def get_theta_hist(self, t, c=None, bin_size_deg=5, spike_mask=None,
                       return_extra=False):
        """Produces an emprical probability distribution of spikes at different
        phases of theta."""
        if 360 % bin_size_deg:
            raise ValueError("bin_size_deg must be a factor of 360.")

        spk_eeg_idx = self.spk_times(t=t, c=c, as_type='e')
        if spike_mask is not None:
            spk_eeg_idx = spk_eeg_idx[spike_mask]

        spk_phase = self.theta_phase[spk_eeg_idx]
        nan_count = np.sum(np.isnan(spk_phase))
        total_count = len(spk_phase)
        spk_phase = spk_phase[~np.isnan(spk_phase)]

        factor = 180. / bin_size_deg / np.pi
        spk_phase_bin_idx = (spk_phase * factor).astype(int)
        n_bins = int(360 / bin_size_deg)

        h = aggregate(spk_phase_bin_idx, 1., func='sum', size=n_bins + 1)
        h[0] += h[-1]  # take care of wrapping/binning issues
        h = h[:-1]
        h /= np.sum(h)  # normalise to sum to 1

        if return_extra:
            return h, np.arange(0, n_bins + 1) * bin_size_deg, nan_count, total_count
        else:
            return h

    """ miscellaneous """

    @append_docstring(get_dir_ratemap)
    def get_dir_r(self, t, c, **kwargs):
        """ uses get_dir_ratemap and then computes the directional R value.         
        Greater than 0.2 is a reasonable level for being classed as directional.
        """
        h, bin_degs = self.get_dir_ratemap(t, c, **kwargs)
        bin_rads = bin_degs / 180. * np.pi
        r = np.abs(np.sum(h * np.exp(1j * bin_rads))) / np.sum(h)

        # for data with known spacing, apply correction factor to correct for bias
        # in the estimation of r (see Zar, p. 601, equ. 26.16 at Matlab circtoolbox)
        # Note this has only a very small effect, and is not really that important
        # if binSize is a constant throughout analysis
        bin_size_rad = bin_rads[1] - bin_rads[0]
        c = bin_size_rad / 2 / np.sin(bin_size_rad / 2)
        r *= c
        return r
