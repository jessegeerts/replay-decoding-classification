# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import rfft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import decimate as sp_decimate

from collections import namedtuple
import matplotlib.pylab as plt
import matplotlib.transforms as mpl_transforms
from numpy_groupies import aggregate_np as aggregate


from utils.custom_functools import append_docstring

from core.trial_basic_analysis import TrialBasicAnalysis as AnalysisClass # used only for docstrings

InfoPowerspectrum = namedtuple('InfoPowerspectrum', 'power freqs s2n')
InfoPowerspectrumExtra = namedtuple('InfoPowerspectrumExtra', ("power freqs "
                        "s2n power_rough freq_search_band peak_freq peak_power peak_mask"))
InfoSpeedFreq = namedtuple('InfoSpeedFreq', ("groups bin_centres bin_widths "
                        "bin_lefts group_medians max_cm_per_sec fit group_is_bad"))
        
class TrialFreqAnalysis(object):
    """ a mixin class for Trial """    
    
    def get_powerspectrum(self, smoothing_hz=0.1875, max_freq=125.0, 
                          sig_to_noise_width=2, return_extra=False):
        """
        Computes fourier power spectrum of eeg signal and smooths with gaussian
        kernel of sigma=smoothing_hz.  It then searches for a peak within
        self.theta_freq_band.
        """
        freq_search_band = self.theta_freq_band

        # mean normalise signal (is this needed?) and zero nans  
        # note this is going to introduce edge effects due to bad sections of eeg              
        eeg = self.eeg(bad_as=None, filetype='eeg')  # FIXME: gotta have better solution than ignoring badness
        eeg = eeg - np.nanmean(eeg) 
        eeg[np.isnan(eeg)] = 0 
        
        nquist_limit = self.eeg_samp_rate / 2
        original_len = len(eeg)
        fft_len = _next_pow2(original_len)
        fft_half_len = int(fft_len / 2 + 1)
        
        # Do the Fourier Transform
        fftRes = rfft(eeg, fft_len)  # the 'r' in 'rfft' means input vector is real not complex
    
        # Get power at all freqs...but symmetry "calzone versus normal pizza"...half but double...
        power = abs(fftRes)**2 # abs^2
        power /= original_len # diving by length gives power density
        power = np.delete(power, range(fft_half_len, len(power))) # cut in half
        power[1:fft_half_len-1]*2 # and times by 2
        
        # Actually we only want freqs up to max_freq...
        freqs = np.linspace(0, 1, fft_half_len) # TODO: this looks ugly could use fft.fftfreq
        freqs = nquist_limit*freqs[freqs <= max_freq]
        power = power[0:len(freqs)]
        
        # do some smoothing
        bins_per_hz = (fft_half_len-1) / nquist_limit
        kernel_sigma = smoothing_hz * bins_per_hz
        sm_power = gaussian_filter1d(power, sigma=kernel_sigma, mode='nearest')
        
        # caculate some metrics
        spectrum_mask_band = (freqs>freq_search_band[0]) & (freqs<freq_search_band[1])
        sm_power_band = sm_power[spectrum_mask_band]
        freqs_band = freqs[spectrum_mask_band]
        max_bin_in_band = np.argmax(sm_power_band)
        band_max_power = sm_power_band[max_bin_in_band]
        freq_at_band_max_power = freqs_band[max_bin_in_band]
        
        # find power in windows around peak, divide by power in rest of spectrum
        # to get signal-to-noise
        spectrum_mask_peak = (freqs>freq_at_band_max_power-sig_to_noise_width/2) \
                           & (freqs < freq_at_band_max_power + sig_to_noise_width/2)
        snr = np.nanmean(sm_power[spectrum_mask_peak]) \
                    / np.nanmean(sm_power[~spectrum_mask_peak]) # TODO: nans, really?
      
        # Return a selection of the stuff from above...
        if not return_extra:
            return InfoPowerspectrum(power=sm_power, freqs=freqs, s2n=snr)
        else:
            return InfoPowerspectrumExtra(power=sm_power, freqs=freqs, s2n=snr, 
                                          power_rough=power, 
                                          freq_search_band=freq_search_band,
                                          peak_freq=freq_at_band_max_power,
                                          peak_power=band_max_power,
                                          peak_mask=spectrum_mask_peak)
    
    @append_docstring(get_powerspectrum)
    def plot_powerspectrum(self,  x_max=25., y_max=None, y_lim_frac= 1.4, **kargs):
        """You can pass kargs through to getPowerspectrum."""
        
        info = self.get_powerspectrum(return_extra=True, **kargs)
        plt.plot(info.freqs, info.power_rough, c=[0.8, 0.8, 0.8],
                 rasterized=True)
        #plt.hold(True)
        plt.plot([info.peak_freq]*2, [0, info.peak_power], 'r-', lw=1,
                 marker='o', rasterized=False)
        plt.plot(info.freqs, info.power, 'k', lw=2, zorder=5, rasterized=False)
        ax = plt.gca()        
        
        # this customTransform alows us to specify x coordinates in data 
        # cordinates and y coordinates in axes coordindates
        custom_trans = mpl_transforms.blended_transform_factory(ax.transData, 
                                                                ax.transAxes)

        plt.fill_between(info.freqs[info.peak_mask], info.power[info.peak_mask],
                         0, color='r', alpha=0.2, zorder=4)
        
        # plot the dashed-blue lines for pre-defined band-range
        plt.plot([info.freq_search_band[0]]*2, [0, 1], 'b--', transform=custom_trans)
        plt.plot([info.freq_search_band[1]]*2, [0, 1], 'b--', transform=custom_trans)                                   
                                       
        plt.xlim(0, x_max)
        if y_max is None:
            plt.ylim(0, info.peak_power*y_lim_frac)
        plt.ylim(0, y_max)
        plt.xticks([0, info.freq_search_band[0], info.freq_search_band[1], x_max])
        plt.yticks([])
        plt.ylabel('power density') # units are roughly watts per Hz
        plt.xlabel('frequency (Hz)')
        #plt.hold(False)
        plt.title('peak @ {:0.1f}Hz | s/n={:0.2f}'.format(info.peak_freq, info.s2n))
        
    @append_docstring(AnalysisClass.speed_bin_inds)
    def get_speed_freq_props(self, **kwargs):
        """
        dropBins should give two values, the number of bins to drop at the low end of
        speed and at the high end of speed.  The bins are still returned at the end
        of the function, but are not used in the fit.
        
        kwargs are passed through to speed_bin_inds()
        """
        speed_bin_idx, (lo_bin, hi_bin), bin_lefts, bin_widths = self.speed_bin_inds(
                                                    return_extra=True, **kwargs)
        
        if self.eeg_samp_rate % self.pos_samp_rate:
            raise NotImplementedError("Cannot downsample from eeg samp rate to pos samp "
                                      "rate because values in Hz are awkward.")
            
        # downsample theta freq to pos rate, and ditch all data corresponding to
        # nans in theta freq (i.e. speed as well as freq)
        down_rate = int(self.eeg_samp_rate / self.pos_samp_rate)
        theta_freq = np.nanmean(self.theta_freq.reshape((-1, down_rate)), axis=1)
        pos_is_bad = np.isnan(theta_freq)
        speed_bin_idx = speed_bin_idx[~pos_is_bad]
        theta_freq = theta_freq[~pos_is_bad]
        
        groups = aggregate(speed_bin_idx, theta_freq, func='array', fill_value=[],
                           size=hi_bin+1)
        group_medians = np.array([np.median(g) for g in groups])
        
        bin_centres = bin_lefts + bin_widths/2.
        group_is_bad = np.isnan(group_medians)
        group_is_bad[[lo_bin, hi_bin]] = True    
                
        fit = np.polyfit(bin_centres[~group_is_bad], group_medians[~group_is_bad], 1)

        return InfoSpeedFreq(groups=groups,
                            bin_centres=bin_centres,
                            bin_widths=bin_widths,
                            bin_lefts=bin_lefts,
                            group_medians=group_medians,
                            max_cm_per_sec=bin_lefts[-1] + bin_widths[-1],
                            group_is_bad=group_is_bad,
                            fit=fit)

    @append_docstring(get_speed_freq_props)
    def plot_speed_freq_props(self, ylims=[6, 12], **kwargs):
        """
        """
        props = self.get_speed_freq_props(**kwargs)
        gca = plt.gca()
        gca.cla()
        group_is_bad = props.group_is_bad
        h = plt.boxplot(props.groups[~group_is_bad], notch=True, 
                        widths=props.bin_widths[~group_is_bad]*0.5,
                        positions=props.bin_centres[~group_is_bad],
                        usermedians=props.group_medians[~group_is_bad])
                        # Note we supply the medians to avoid forcing the boxplot to recompute them
                        
        #gca.hold(True)
        max_speed = props.max_cm_per_sec
        first_last_idx = (~group_is_bad).nonzero()[0][[0,-1]]
        xlim = props.bin_lefts[first_last_idx] \
                + np.array([0, props.bin_widths[first_last_idx[1]]])
                
        plt.plot(xlim,[props.fit[1] + xlim[0]*props.fit[0],
                       props.fit[1] + xlim[1]*props.fit[0]], 'b')
        plt.setp(h['whiskers'], c=[0.8, 0.8, 0.8], lw=2, linestyle='-')
        plt.setp(h['boxes'], c='k')
        plt.setp(h['fliers'], c=[0.6, 0.6, 0.6])
        plt.xlabel('speed (cm/s)')
        plt.ylabel('theta freq (Hz)')
        plt.ylim(ylims)
        plt.xlim([0, max_speed])
        plt.xticks(props.bin_centres[~group_is_bad])
        plt.title('[{:0.1f} + {:0.4f}cm/s] Hz'.format(props.fit[1], props.fit[0]))

    


def _next_pow2(a):
    a = int(a)
    b = 1 << a.bit_length()
    return b >> (b>>1==a) # "if a was already a pwoer of 2, then divide b by 2"