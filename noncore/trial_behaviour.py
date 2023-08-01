# -*- coding: utf-8 -*-
import numpy as np
import collections
import matplotlib.pylab as plt

from utils.custom_functools import append_docstring
from numpy_groupies import aggregate_np as aggregate


BehaviourProps = collections.namedtuple('BehaviourProps',(
        "dir_stereotyping mean_speed area_covered active_fraction median_speed"
        " upper_quartile_speed lower_quartile_speed"))

class TrialBehaviour(object):
    """ mixin class for Trial, adding functions for analysing behaviour """
    
    def get_dir_variance_map(self, bin_size_cm=5.0, nodwell_mode='ma',
                             force_use_disp=False):
        """Computes the directional variance of position sampling in each
        spatial bin, i.e. this has nothign to do with spikes at all. 
        If the behaviour is stereotyped the variance will be especially low
        in all bins..so you can take the mean across bins to get an estimate
        of "stereotyping".        
        TODO: you may find you want to mask on speed > some_threshold.
        """
        xy_idx, w, h = self.xy_bin_inds(bin_size_cm=bin_size_cm)
        dir_ = self.dir_disp if force_use_disp else self.dir
        dir_data = np.exp(1j*dir_)
        complex_means = aggregate(xy_idx[::-1], dir_data, func='mean', 
                                  size=(h, w), fill_value=np.nan)
        counts = aggregate(xy_idx[::-1], 1, func='sum', size=(h, w))
        ret = 1 - np.abs(complex_means)
        
        if nodwell_mode.lower() == 'ma':
            return np.ma.array(ret, mask=counts==0)
        elif nodwell_mode.lower() == 'nan':
            return ret

    @append_docstring(get_dir_variance_map)
    def plot_dir_variance_map(self, **kwargs):
        vm = self.get_dir_variance_map(**kwargs)
        self.plot_spa_ratemap(made_earlier=vm)
        plt.title('directional sampling variance')
        plt.colorbar()
        
    @append_docstring(get_dir_variance_map)       
    def get_behaviour_props(self, dwell_time_thresh_sec=0.25, 
                          speed_active_thresh=5.0, **kwargs):
        """
        Also need a better way of passing kwargs to different functions.
        
        ``dwell_time_thresh_sec`` - we threshold the dwell map, post smoothing, 
        and count the number of bins with dwell time greater than this.
        the ``area_covered`` is expressed in m2.
        
        ``speed_active_thresh`` is a speed in cm/s, used to compute the
        fraction of the trial spent active, ``active_fraction``. It is also
        used to compute the ``mean_active_duration``, where ...?
    
        TODO: add more stuff.
        """        
        vm = self.get_dir_variance_map(nodwell_mode='nan',**kwargs)
        rm, bin_size_cm = self.get_spa_ratemap(**kwargs)
        area = np.sum(rm > dwell_time_thresh_sec) * (bin_size_cm/100)**2
        speed = self.speed
        sp_lq, sp_med, sp_uq = np.percentile(speed, [25, 50, 75])
        return BehaviourProps(dir_stereotyping=np.nanmean(vm),
                               mean_speed=np.mean(speed),
                               median_speed=sp_med,
                               lower_quartile_speed=sp_lq,
                               upper_quartile_speed=sp_uq,
                               active_fraction=np.mean(speed > speed_active_thresh),
                               area_covered=area)
        
        

    
 