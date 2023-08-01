# -*- coding: utf-8 -*-
from __future__ import division # it turns out that we are relying on this somewhere 

import numpy as np
from numpy import newaxis as _ax
import matplotlib.pylab as plt

from collections import namedtuple
from utils.custom_np import ignore_divide_by_zero
from utils.custom_functools import append_docstring

InfoIdealGrid = namedtuple('InfoIdealGrid', "orientation_deg scale_cm offset_x offset_y pearson")

# the main affine transformation to go to/from orthogonal to/from 60-degree axes
_shear = np.array([[1, 0], [-1/np.sqrt(3), 2/np.sqrt(3)]])
_shear_inv = np.array([[1, 0 ], [ 1./2, np.sqrt(3)/2]])

def _make_rot_mat(theta):
    """simply use -theta to invert."""
    return np.array([[np.cos(-theta), -np.sin(-theta)],
                     [np.sin(-theta), np.cos(-theta)]])
    
def _build_ideal_unit_grid(n_bins_side):
    """Creates one unit of an ideal grid of shape=[n_bins_side, n_bins_side]
      note that the 2 dimesnions of the output array correspond to two axes
      of the grid, i.e. not to the usual cartesian axes - that is why they
      bump looks skewed when matshow'n.
    """
    shear_inv = _shear_inv
    
    xy = np.meshgrid(*[(np.arange(n_bins_side) + 0.5)/float(n_bins_side)]*2)
    xy = np.vstack([v.ravel() for v in xy])
    
    xy = np.dot(xy.T, shear_inv).T
    
    omega_theta = np.array([-np.pi/6, np.pi/6, 3*np.pi/6]) # Angle of bands relative to one another
    omega_radius = 1./np.cos(30/180.*np.pi) * 2 * np.pi # Convert to grid scale and get appropriate vector
    omega = np.vstack((omega_radius * np.cos(omega_theta), 
                       omega_radius * np.sin(omega_theta)))
    ideal_unit_grid = np.sum(np.cos(np.dot(omega.T, xy)), axis=0)
    ideal_unit_grid[ideal_unit_grid<=0] = 0
    ideal_unit_grid.shape = n_bins_side, n_bins_side
    ideal_unit_grid /= np.mean(ideal_unit_grid)
    return ideal_unit_grid
    

def _next_pow2(a):
    a = int(a)
    b = 1 << a.bit_length()
    return b >> (b>>1==a) # "if a was already a pwoer of 2, then divide b by 2"


class TrialIdealGrid(object):
    
    def get_fitted_ideal_grid(self, t, c, scales_list_cm=np.linspace(28, 58, num=20),
                               theta_list_deg=np.linspace(0, 60, num=60, endpoint=False),
                               n_bins_side=32):
        """
        This is roughly the same as the process described in the following paper:
            `"Grid Cells Form a Global Representation of Connected
            Environments", -Carpenter, .., Burgess, Barry, 2015.`
        Note however that this is slightly different in a few ways - see notes 
        below...
        
        For a range of angles and scales, this bins up the raw pos and spike
        data into a unit grid ratemap of shape=[n_bins_side, n_bins_side].
        
        This involves a rotation, a shear, and a modulo operation.
        No smoothing is done before doing spikes/dwell, and we assume that all 
        bins in the ratemap have dwell>0. Also, because (a/b)+(c/d) != (a+c)/(b+d),
        the ratemaps constructed here are slightly different to what you get by 
        going via a normal ratemap.  Perhaps it would be worth rewriting this
        somewhat to go via a normal ratemap, (with fairly small bins), which you
        could smooth in the normal way.  You would then apply the rotations, shearings
        to the bin centres and accumulate *means* using bincounts(bin_idx, weights=rm).
        This would probably be even faster, come to think of it, and it would no
        longer scale with n_pos.  You could also cache the bin_inds and normalisation
        for the mean-finding, since those things are independant of the ratemap's 
        content, only its dimensions.
    
        The unit ratemap is cross-correlated with an ideal unit ratemap, in order
        to find the best phase offset (vector) for the given rotation and scale.
        
        The overal best rotation, scale and phase offset are returned.
        
        Note that this thing is fast* due to a number of optimisation strategies!!
        However it may be almost equivalent to the extended form of phase 
        correlation for image registration, which I think would be a lot faster
        still.
        
        Note that for various reasons, this function makes its own ratemaps
        rather than relying on the basic analysis function.
        
        *DM: on my laptop it takes about 1.4s to run on 20mins of trial at 50Hz,
        with 2.5k spikes, searching 20scales, 60angles, and 32x32 offsets.
        """
        if _next_pow2(n_bins_side) != n_bins_side or n_bins_side.bit_length() > 8:
            raise ValueError("n_bins_side must be power of 2, maximum 256.") # so we can do 2d indexing in uint16s
    
        shear = _shear
        
        # could cache these if you really wanted to
        ideal = _build_ideal_unit_grid(n_bins_side)
        ideal_fft = np.fft.fftn(ideal, axes=(0, 1))
        sum_ideal = np.sum(ideal)
        sum_ideal_sqrd = np.sum(ideal**2)
        
        absolute_shift = 10000
        
        xy = self.xy
        spk_pos_idx = self.tet_times(t, c, as_type='p')
        
        info = InfoIdealGrid(0, 0, 0, 0, -1)
        p_max = -1
    
        scales_list_cm = scales_list_cm[:, _ax, _ax]
        with ignore_divide_by_zero():
            for theta_deg in theta_list_deg:
                theta = theta_deg/180.*np.pi 
                rot = _make_rot_mat(theta)
                
                # convert raw xy values to bins in the unit tile...
                # or rather, boradcast out xy to give bins in the unit tile for
                # each requested scale            
                xy_rotated_and_sheared = np.dot(xy.T, np.dot(rot, shear)).T
                xy_rotated_and_sheared += absolute_shift # this ensures we are in positive domain
                xy_bin_idx = np.empty((len(scales_list_cm),) + xy.shape, dtype=np.uint16) # used as out= on next line...
                np.multiply(xy_rotated_and_sheared[_ax, :, :], 
                            float(n_bins_side)/scales_list_cm,
                            out=xy_bin_idx, casting='unsafe') # using the out param we get .astype(int), i.e. floor, all in one op (well, probably not really: see np.multiply.types)
                xy_bin_idx &= n_bins_side - 1 # equivalent to %= n_bins_side for pows of 2
                
                # np.ravel_multi_index, but faster given pow of 2, and in-place...
                xy_bin_idx[:, 0] <<= (n_bins_side-1).bit_length() 
                xy_bin_idx[:, 0] |= xy_bin_idx[:, 1] 
                xy_bin_idx = xy_bin_idx[:, 0]
                
                # for each scale, bin up dwell and spike and divide to get ratemap...
                rm = np.empty((n_bins_side, n_bins_side, len(scales_list_cm)))
                for ii, xy_bin_idx_ii in enumerate(xy_bin_idx):
                    dwell = np.bincount(xy_bin_idx_ii, minlength=n_bins_side**2)
                    spike = np.bincount(xy_bin_idx_ii[spk_pos_idx], minlength=n_bins_side**2)        
                    rm[..., ii] = (spike / dwell).reshape((n_bins_side, n_bins_side))
            
                # do convolution of stack of ratemaps with unit ideal grid...
                rm[np.isnan(rm)] = 0 # this is a hack, to avoid empty dwells
                rm_fft = np.fft.fftn(rm, axes=(0, 1))    
                rm_x_ideal = np.fft.ifftn(rm_fft * ideal_fft[..., _ax],
                                            axes=(0, 1)).real
                sum_rm = np.sum(rm, axis=(0, 1), keepdims=True)
                sum_rm_sqrd = np.sum(rm**2, axis=(0, 1), keepdims=True)
                
                # turn convolution into correlation...
                n = n_bins_side*n_bins_side
                p = (rm_x_ideal*n - sum_rm * sum_ideal) / \
                        (np.sqrt(n*sum_rm_sqrd - sum_rm**2) * np.sqrt(n*sum_ideal_sqrd - sum_ideal**2))
    
                # find max, and possibly update max over all angles...
                new_argmax = np.unravel_index(np.nanargmax(p), p.shape) # TODO: check that nans are from near-zero complex fft issue only
                new_max = p[new_argmax]
                if new_max > p_max:
                    p_max = new_max
                    p_scale = np.asscalar(scales_list_cm[new_argmax[2]])
                    info = InfoIdealGrid(orientation_deg=theta_deg,
                                          scale_cm=p_scale,
                                          offset_x=(new_argmax[1]/n_bins_side - absolute_shift % p_scale) % 1, # TODO: this might be wrong!!
                                          offset_y=(new_argmax[0]/n_bins_side - absolute_shift % p_scale) % 1,
                                          pearson=p_max)
                # if you know the true scale, you can view the correlation at various angles for debugging
                #matshow(p[...,16], vmin=-1, vmax=1, fignum=4); plt.title("theta: {:0.1}".format(theta/np.pi*180))       
                #plt.pause(0.1)
            return info          
    
    @append_docstring(get_fitted_ideal_grid)
    def plot_fitted_ideal_grid(self, t, c, made_earlier=None, **kwargs):
        """
        uses get_fitted_ideal_grid and then shows the results as circles
        on top of a normal ratemap.     
        
        TODO: Fix interpretation of phases..I'm pretty sure they're not correct
        currently but it's a pain to sort it out... see offset_x/y assigment
        in get_* above as well as rendering code below.
        """
        if made_earlier is None:
            info = self.get_fitted_ideal_grid(t, c, **kwargs)
        else:
            info = made_earlier

        # "forward" transform the bounding box of the environment (expanding by
        # the radius of the circles we are going to plot, because circle centres
        # beyond the edge of the environment can still show their perimiter).
        # the trasnformed box gives coordinates in grid space.
        r = info.scale_cm/3
        box = np.array([[-r, self.w + r, -r, self.w + r],
                        [self.h + r, -r, -r, self.h + r]])
        theta = info.orientation_deg/180.*np.pi 
        rot = _make_rot_mat(theta)
        box_transformed = np.dot(box.T, np.dot(rot, _shear)).T / info.scale_cm \
                          - np.array([info.offset_x, info.offset_y])[:, _ax]        
        min_x, min_y = np.min(box_transformed, axis=1)     
        max_x, max_y = np.max(box_transformed, axis=1)     
        
        # now we have a reasonable bound on the range of x and y values in grid space,
        # so we can inverse transform that to get back to ratemap space
        rot_inv = _make_rot_mat(-theta)
        xy = np.meshgrid(np.arange(np.floor(min_x), np.ceil(max_x)+1) + info.offset_x,
                         np.arange(np.floor(min_y), np.ceil(max_y)+1) + info.offset_y)        
        xy = np.vstack([p.ravel() for p in xy])
        xy = np.dot(xy.T * info.scale_cm, np.dot(_shear_inv, rot_inv)).T

        # finally, actually plot the ratemap and show circles...        
        self.plot_spa_ratemap(t, c) 
        ax = plt.gca()
        if self.pos_shape is not None:
            clip_region = self.pos_shape.clip_region()
            ax.add_patch(clip_region) # you need to do this in order to use it for clipping
        for circ_xy in xy.T:
             patch = plt.Circle(circ_xy, radius=r, zorder=1, fill=False, lw=4)
             ax.add_patch(patch)
             if self.pos_shape is not None:
                 patch.set_clip_path(clip_region)


        plt.title(("{orientation_deg:0.1f}deg {scale_cm:0.1f}cm "
                   "({offset_x:0.2f}, {offset_y:0.2f}) | p={pearson:0.2f}").format(**info._asdict()))


