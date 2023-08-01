# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import scipy.ndimage.filters as sp_filters
import numpy as np
from numpy import newaxis as _ax
from collections import namedtuple

from mpl_toolkits.axes_grid1 import make_axes_locatable 
from utils.custom_functools import append_docstring
from utils.custom_mpl import matshow_vf


WarpInfo = namedtuple('WarpInfo', ("argmax max mean_of_mags mag_of_mean_vec "
                                    "weighted_dir_var"))
                                    
class TrialWarp(object):
    """mixin class for computing "warp" between two ratemaps."
        dependency: trial_gc (in addition to usual dependencies)."""
        
    def get_spa_warp(self, t, c, bin_size_cm=4, max_shift=6, kernel_width=8,
                     return_extra=False, **kwargs):
        """
        ``max_shift=W``.  shift ``-W`` to ``+W``, total ``(2W+1)*(2W+1)``.
        
        Splits trial into two halves, computing a ratemap for each half.
        Then returns a map giving the warp needed to locally maximise correlations.
        
        The warp is found by explicitly looping over a grid of potential offsets
        of half-2 relative to half-1.
        
        For each warp, a gaussian weighted bump of points is considered for half-1's
        ratemap and the corresponding bump of points for half-2's ratemap.
        The shift with the greatest correlation is recorded in the output using
        complex numbers to capture the x and y shift.
        
        Note this analysis was created by DM and has not yet been successfully used
        anywhere.
        """
        
        A, B, _, _ = self.get_spa_stability(t, c, bin_size_cm=bin_size_cm,
                                         return_extra=True, **kwargs)
        
        A_ones = ~A.mask
        B_ones = ~B.mask
        A = A.filled(fill_value=0) # zeros don't contribute to sums, and we normalise using the *_ones, so it's ok.
        B = B.filled(fill_value=0)
        A2 = A**2
        B2 = B**2
        
        W = max_shift # abbreviate
    
        # a weighted sum is more commonly refered to as convolution...
        def kernel_sum(X, ):
            return sp_filters.gaussian_filter(X, kernel_width, mode='constant')
    
        if return_extra:
            C_sub_list = []
            
        C_max = np.full(A.shape, -np.Inf)
        C_argmax = np.zeros(A.shape, dtype=complex)
        
        # Loop over a grid of (x, y)-shifts, computing correlations using a kernel centred at each bin.
        for y_shift in range(-W, W+1):
            for x_shift in range(-W, W+1):
                
                # for the current (x, y)-shift, get the slices into A and B
                slc_A = (slice(max(0, y_shift) , y_shift if y_shift < 0 else None) ,
                          slice(max(0, x_shift) , x_shift if x_shift < 0 else None) )
                slc_B = (slice(max(0, -y_shift) , -y_shift if y_shift > 0 else None), 
                          slice(max(0, -x_shift) , -x_shift if x_shift > 0 else None))
                
                # use those slices to actually select A, A-squared, A_ones, and same for B
                B_ones_sub = B_ones[slc_B]
                A_ones_sub = A_ones[slc_A]
                A_sub = A[slc_A].copy()
                B_sub = B[slc_B].copy()        
                A2_sub = A2[slc_A].copy()
                B2_sub = B2[slc_B].copy()
         
                # We want to match nodwell for A and B on this particular shift
                AB_ones_sub = A_ones_sub & B_ones_sub                
                A_sub[~B_ones_sub] = 0
                B_sub[~A_ones_sub] = 0
                A2_sub[~B_ones_sub] = 0
                B2_sub[~A_ones_sub] = 0
          
                # compute weighted sums using kernel centred on each bin
                AB_ones_sub = kernel_sum(AB_ones_sub.astype(np.float32))       
                AB_sub = kernel_sum(A_sub * B_sub)
                A_sub = kernel_sum(A_sub)
                B_sub = kernel_sum(B_sub)        
                A2_sub = kernel_sum(A2_sub)
                B2_sub = kernel_sum(B2_sub)       
                
                # Use weighted sums to compute pearson correlation over kernel centred at each bin
                C_sub = (AB_sub * AB_ones_sub - A_sub*B_sub) / \
                    (np.sqrt(A2_sub-A_sub**2) * np.sqrt(B2_sub-B_sub**2) )
                
                # If correlation for this shift is better than previous shifts, then
                # record the better correlation value and record this shift as _argmax.
                is_better = np.zeros(A.shape, dtype=bool)
                is_better[slc_A] = is_better_sub = C_sub > C_max[slc_A]        
                C_max[is_better] = C_sub[is_better_sub]
                C_argmax[is_better] = x_shift + 1j*y_shift
    
                # Normally we don't care about the correlations once we've done the max/argmax
                if return_extra:
                    C_sub_list.append(C_sub)
          
        # Mask bins outside of the posShape
        shape_mask = self.pos_shape.make_mask(self.w, self.h, bin_size_cm)
        if shape_mask is not None:
            C_argmax[shape_mask] = np.nan
            C_max[shape_mask] = np.nan
        
        if not return_extra:
            return C_argmax, C_max, W
        else:
            return C_argmax, C_max, W, C_sub_list, A, B, bin_size_cm
  
    @append_docstring(get_spa_warp)          
    def get_spa_warp_props(self, t=None, c=None, bin_size_cm=4, **kwargs):
        C_argmax, C_max, W = self.get_spa_warp(t, c, bin_size_cm=bin_size_cm,
                                               **kwargs)
        non_nan_warp = C_argmax[~np.isnan(C_argmax)]
        mean_of_mags = np.mean(np.abs(non_nan_warp)) * bin_size_cm
        mag_of_mean_vec = (np.abs(np.sum(non_nan_warp.real)
                               + 1j*np.sum(non_nan_warp.imag))) / len(non_nan_warp) * bin_size_cm
        weighted_dir_var = 1 - mag_of_mean_vec / mean_of_mags # Note: we've essentially cancled times-n on top and bottom
        return WarpInfo(argmax=C_argmax,
                        max=C_max,
                        mean_of_mags=mean_of_mags,
                        mag_of_mean_vec=mag_of_mean_vec,
                        weighted_dir_var=weighted_dir_var 
                        )
         
    @append_docstring(get_spa_warp)
    def plot_spa_warp(self, t=None, c=None, warp=None, full_plot=False, 
                      show_key=True, **kwargs):
        """
        When ``full_plot`` is true, this plots/clears figure 1 and 2.
        
        you  can provide the output of a pre-computed warp or just
        provide args to pipe through to get_spa_warp.
        """
        if warp is None:
            warp = self.get_spa_warp(t, c, return_extra=full_plot, **kwargs)
    
        if len(warp) > 3:        
            C_argmax, C_max, W, C_sub_list, A, B, bin_size_cm = warp
        else:
            C_argmax, C_max, W = warp
    
            
        if full_plot:
            plt.figure(1)
            plt.clf()
            for ii, C in enumerate(C_sub_list):
                plt.subplot(W*2+1, W*2+1, ii+1)
                plt.imshow(C, vmin=-1, vmax=1, interpolation='nearest')        
                
            plt.figure(2)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.imshow(A)
            plt.title("ratemap half-1")
            
            plt.subplot(2, 2, 2)
            plt.imshow(B)
            plt.title("ratemap half-2")
        
            plt.subplot(2, 2, 3)
            plt.imshow(C_max, vmin=-1, vmax=1)
            plt.title("max correlation (over shifts)")
            plt.colorbar()
            
            plt.subplot(2, 2, 4)
        
        plt.title("argmax correlation (over shifts)")
        self.plot_spa_ratemap(made_earlier=C_argmax, vmax=np.sqrt(W**2 + W**2))
        if show_key:
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="15%", pad=0.05)
            key = np.arange(-W, W+1)[_ax, :] + np.arange(-W, W+1)[:, _ax]*1j
            matshow_vf(key, vmax=np.sqrt(W**2 + W**2))
            cax.axis('off')

        
        
    
