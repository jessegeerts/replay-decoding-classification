# -*- coding: utf-8 -*-
import os
from warnings import warn
import numpy as np
from scipy.io import loadmat

class TrialMoserMat(object):
    """
    This implements the `Trial` class interface as described in the main readme.
    Specially it is intended for loading the .mat files made available by the
    Moser lab.
    TODO: have a proper PosProcessor a bit like TrialAxona.
    """    
    def __init__(self, fn, pos_tol=0.001, pos_desired_sample_rate=50.0):
        self._fn = fn
        self._pos_tol = pos_tol
        self._pos_desired_sample_rate = pos_desired_sample_rate
        
    def _load_pos_into_cache(self):
        """
        We expect a mat file with an array called post giving times in seconds.
        We return a bare-bones header and the xy data.
        
        This has to be sent through a special pos_processor.
        
        ``desired_sample_rate`` is only used when the sample rate is not completely
        uniform (meaning the expected time values never deviate by more than ``tol``).
        In this non-uniform case we interpolate x and y data to make it seem like the sample
        rate was uniform, using this rate.
        """
        if not self._cache_has("pos", "_cache_xy _cache_duration".split(" ")):
            M = loadmat(self._fn + "_POS.mat")    
            post = M['post'].squeeze()
            
            # get pos inverse sample rate, and duration
            dt = np.diff(post)
            mean_dt = np.mean(dt)
            self._cache_duration = post[-1] + mean_dt        
                
            #if we are missing the first sample then we need to add it as nan
            #this is probably because the trial splitting code used a greater than rather than >= ..I'm guerssing.
            if 0.5* mean_dt < post[0] < 1.5 *mean_dt:
                for key, val in M.iteritems():
                    if key in ('posx','posy','posx2','posy2','post'):
                        M[key] = np.insert(val.squeeze(), 0, np.nan)
                post = M['post'] #get corrected version
            
            # check pos times correspond to unifronm steps 
            if any(abs(np.arange(0,len(post))*mean_dt-post) > self._pos_tol):
                warn("interpolating pos to uniform times")
                
            # interpolate values to produce an artifically uniform sample rate
            # and if there were any nans in the xy data we also interp across them 
            # at this point
            # TODO: for uniformly sampled data should wait for proper post processing
            mean_dt = 1.0/self._pos_desired_sample_rate#this is our artifically imposed inverse sample rate
            new_post = np.arange(0, self._cache_duration-mean_dt*0.5, mean_dt)
            for key, val in M.iteritems():
                if key in ('posx','posy','posx2','posy2'):
                    val = val.squeeze()
                    if val.size:
                        M[key] = np.interp(new_post, post[~np.isnan(val)],
                                        val[~np.isnan(val)])
                    
            self._cache_xy = np.vstack((M['posx'].squeeze(), M['posy'].squeeze()))
            self._cache_xy -= np.min(self._cache_xy, axis=1, keepdims=True)
            self._cache_xy.setflags(write=False)
            
            warn('Moser pos-post-processing not properly implementeted')
            # TODO: use posx2 and proper post processing including shape       
            #if 'posx2' in M:
            #    xy2 = vstack((M['posx2'].squeeze(),M['posy2'].squeeze()))
                    
    def _load_eeg_into_cache(self):
        if not self._cache_has("eeg", ("_cache_eeg", "_cache_eeg_samp_rate")):            
            M = loadmat(self._fn + "_EEG.mat")
            self._cache_eeg_samp_rate = float(np.asscalar(M.get('fs', M['Fs'])))
            eeg = M['EEG'].squeeze()
            eeg.setflags(write=False)            
            self._cache_eeg = eeg
        
    def _load_tc_into_cache(self, t, c):
        if not self._cache_has("tet", "_cache_tet_times", (t,c)):
            M = loadmat("{base:}_T{tet:}C{cell:}.mat".format(base=self._fn,
                        tet=t, cell=c))
            if 'ts' in M:
                times = M['ts'].squeeze()
            elif 'cellTS' in M:
                times = M['cellTS'].squeeze()
            elif 'cellTs' in M:
                times = M['cellTs'].squeeze()
            else:
                raise Exception("Could not find data in mat file. keys are: " + ",".join(M.keys()))
            times.setflags(write=False)
            self._cache_tet_times[(t,c)] = times
        
    @property
    def experiment_name(self):
        """ returns a string, giving the experiment name."""
        return os.path.split(self._fn)[-1]

    @property
    def duration(self):
        """ returns the experiment duration in seconds."""
        self._load_pos_into_cache()
        return self._cache_duration

    def __str__(self):
        return "<TrialMoserMat instance: fn='" + self._fn + "'>"

    def __repr__(self):
        return str(self)
        
    """ Position related stuff """

    @property
    def n_pos(self):
        """returns number of pos samples"""
        self._load_pos_into_cache()
        return self._cache_xy.shape[1]

    @property
    def pos_samp_rate(self):
        """returns position sampling rate in Hz"""
        return self._pos_desired_sample_rate

    @property
    def xy(self):
        """returns [2xn_pos] values in centimeters, x values are [0,w], 
        y values are [0, h]."""
        self._load_pos_into_cache()
        return self._cache_xy

    @property
    def w(self):
        """ returns arena width in cm"""
        return np.max(self.xy[0]) #TODO: implement properly

    @property
    def h(self):
        """ returns arena height in cm"""
        return np.max(self.xy[1]) #TODO: implement properly
        
    @property
    def pos_shape(self):
        """returns an instance of a class implementing the PosShape interface."""
        return DummyPosShape()

    @property
    def speed(self):
        """returns values in centimeters per second, lem=n_pos"""
        if not self._cache_has('pos','_cache_speed'):
            diff_xy = np.diff(self.xy, axis=1)
            if len(diff_xy):
                self._cache_speed = np.hypot(diff_xy[0], diff_xy[1])
                self._cache_speed = np.append(self._cache_speed,[0])
                self._cache_speed = self._cache_speed * self.pos_samp_rate
            else:
                self._cache_speed = np.array([])
            self._cache_speed.setflags(write=False)
        return self._cache_speed

    @property
    def dir(self):
        """ returns direction of facing in radians len=n_pos,
        i.e. using all availalbe LEDs not displacement vector"""
        raise NotImplementedError

    @property
    def dir_disp(self):
        """returns direction of displacement vector in radians, len=n_pos"""
        raise NotImplementedError        

    """ Spike-related stuff """

    def tet_times(self, t, c, as_type='s'):
        """ returns the spikes times for the given tetrode/cell.  as_type has the
          following possible values: 'x' - use original timebase; 's' - use seconds;
          'p' - use index into pos samples; 'e' - use index into eeg samples. 
          Note that the two indices return ints, but the seconds are doubles. """
        self._load_tc_into_cache(t, c)
        times = self._cache_tet_times[(t,c)]
        print(times[:10], "...")
        raise NotImplementedError("what is the timebase? see times pritned above")

    """ EEG-related stuff """

    @property
    def eeg_samp_rate(self):
        """ value in Hz. """
        self._load_eeg_into_cache()
        return self._cache_eeg_samp_rate

    def eeg(self, bad_as='nan'):
        """ gives the eeg values in volts. """
        self._load_eeg_into_cache()
        return self._cache_eeg        

    @property
    def eeg_is_bad(self):
        return None # it's all fine, obviously


class DummyPosShape(object):
    def is_outside(self, x, y):
        """For a point, (x,y) in cm, it returns True if the point is outside
        the shape, otherwise False.        """
        return False

    def make_mask(self, w, h, bin_size_cm, inner_boundary_cm=None):
        """Produces a "ratemap-like" boolean mask, which is True outside the
         shape and False inside.  However, when inner_boundary is a positive
         number or sequence of positive numbers, we create a label matrix not 
         a mask.  The region outside the shape is labeled 0, and inside the 
         shape is labeled from 1 to n as you go deeper inside the interior."""
        return None # no mask

    def plot(self):
        """Plots the shape using cm coordinates."""
        pass # implementation optional