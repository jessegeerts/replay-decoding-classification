# -*- coding: utf-8 -*-

import numpy as np
from numpy import newaxis as _ax

import numpy_groupies as npg
import matplotlib.pylab as plt
from scipy.special import factorial as sp_factorial
from scipy.ndimage import grey_dilation
from collections import namedtuple
from utils.custom_functools import append_docstring

                          
InfoRunDrift = namedtuple('InfoRunDrift', ("pos_run_label likelihood S "
                                        "xy_bin_inds spike_counts d0 d1 rm "
                                        "mle_height"))
MLEDriftProps = namedtuple('MLEDriftProps', ("run_mean_drift run_std_drift "
                                        "run_drift_likelihood_mean "
                                        "run_drift_likelihood_std"))
        

class TrialMLEDrift(object):
    """ this is a mixin for Trial. Dependends on trial_gc """
    
    def get_gc_run_labels(self, t, c, mode_fields=3, min_run_duration=0.25):
        """
        ``min_run_duration`` is expresed in seconds. Any runs shorter than this
        are discarded at the start.
        
        ``mode_fields`` is passed through to get_field_label_at_xy.
        
        TODO: this should probably be moved to trial_gc.py
        """
        pos_field_label = self.get_field_label_at_xy(t, c, mode='pos',
                                                     mode_fields=mode_fields)
        pos_run_label = npg.label_contiguous_1d(pos_field_label)
        run_duration = npg.aggregate_np(pos_run_label, 1)
        if min_run_duration:
            pos_run_label = npg.relabel_groups_masked(pos_run_label,
                                                      run_duration >= min_run_duration*self.pos_samp_rate)
        return pos_run_label
        
    @append_docstring(get_gc_run_labels)
    def get_mle_drift_info(self, t, c,  bin_size_cm=2.0, min_run_duration=0.25, #seconds
                          max_drift=8, mode_fields=3, return_extra=False,
                          mode_resultant='peak', split='field runs',
                          pos_run_label=None):
        """
    
        ``max_drift``, aka ``S`` defines the grid on which we evaluate lieklihoods.
        Grid is ``-S:S x -S:S ``   (":S" here includes S).This is expressed in bins.
    
        ``mode_fields`` and ``min_run_duration`` are passed through to
        ``self.get_gc_run_labels``. Alternatively you can provide pos_run_label
        here, which is of the same form as the output produced by that function.
    
    
        This function is clsoely inspired by Caswell's MLE stuff (which I guess
        isn't really his at all), although the actual motivation was to get
        an estimate of the *distance* from the peak of a run to the trial-average
        peak of the field on the ratemap.  The reason we ended up with this is that
        a naive alogrithm which simply finds the location of the peak on a run and
        takes the distance to the field peak will have a strange dependance on 
        the shape and location of the run relative to the field centre...which you
        could deal with using some sort of transformation (a la Jeewajee 2014), but
        that can get messy unless the runs are particularly simple straight lines.
        Here we actually do what you really want, which is to ask: "when the rat makes
        this little segment of run, which bit of the ratemap does it think it is
        running across? And how far is this from the true location on the ratemap."
        So, we simply evalutate the likelihood of oversving the spike train at a grid
        of possible offsets, centred on zero-drift (0, 0).  The likelihood is a basic
        Poisson calculation, reading lambda from the ratemap, and taking k as simply
        the spike counts in a pos samp time bin (i.e. 0.02s). The total likelihood
        of a given offset is the product of the Poission values evaluated at each
        pos sample on the run.  The likelihood is computed in log-space, as products
        then become sums and accuraccy is improved.
        
        Once we have the likelihood map (which you might want to call the "posterior")
        we simply take the centre of mass and claim that is the drift for the run.
        
        Roughly speaking, sections of run with no spikes are "pushed" out of the
        field, and sections with lots of spikes are "push" into the field..it's a
        bit like molecules with hydrophilic/phobic bits if you want a perversely
        non-intuative analogy!
        
        ``mode_resultant`` can be:    
            * ``'peak'`` - find the peak closest to the centre
            * ``'mean'`` - take a weighted centre-of-mass thing. 
        
        """
        S = max_drift # alias to short name
        
        spike_counts = self.get_t_rate(t, c, smoothing_type=None, as_count=True)
        if pos_run_label is None:
            pos_run_label = self.get_gc_run_labels(t,c, mode_fields=mode_fields,
                                                    min_run_duration=min_run_duration)
        xy_bin_inds, w, h = self.xy_bin_inds(bin_size_cm=bin_size_cm)
        xy_bin_inds = xy_bin_inds[::-1] # confusingly x, y are returned in the "wrong" order
        
        # we only care about runs here, forget everything else...
        is_run = pos_run_label.astype(bool)
        spike_counts = spike_counts[is_run].astype(int)
        pos_run_label = pos_run_label[is_run] - 1 # we can now use 0 as a run label not an outside-the-field label
        xy_bin_inds = xy_bin_inds[:, is_run]
        
        # Note that interpolation is important: nodwell zeros mess things up later on
        # note also that we don't convert to Hz.
        rm, _ = self.get_spa_ratemap(t, c, bin_size_cm=bin_size_cm,
                                    smoothing_type='gaussian', smoothing_bins=3,
                                    nodwell_mode='interp', norm='pos')  
        log_rm  = np.log(rm) # logs are slow, only compute them once please
        
        drift_vals = np.arange(-S, S+1) # we use this in x and y
    
        # from the rm, we are going to read an [nPos x n_X_drift x n_Y_dift] array of values 
        bin_ind_0 = xy_bin_inds[0][:, _ax, _ax] + drift_vals[_ax, :, _ax]    
        bin_ind_1 = xy_bin_inds[1][:, _ax, _ax] + drift_vals[_ax, _ax, :]
        bin_ind_0 = np.clip(bin_ind_0, 0, w-1)    
        bin_ind_1 = np.clip(bin_ind_1, 0, h-1)
        
        
        # We compute log likelihood, which consits of three summation terms        
        lambda_ = rm[bin_ind_0, bin_ind_1] 
        log_lambda = log_rm[bin_ind_0, bin_ind_1]
        
        log_factorial_lookup = np.log(sp_factorial(np.arange(10))) # logs and factorials are slow, so only compute once
        
        sum_log_k_factorial = npg.aggregate_np(pos_run_label,
                                               log_factorial_lookup[spike_counts])[:, _ax, _ax]
        sum_lambda = npg.aggregate_np(pos_run_label, lambda_, axis=0)
        
        sum_k_log_lambda = npg.aggregate_np(pos_run_label, 
                                            spike_counts[:, _ax, _ax] * log_lambda,
                                            axis=0)
        LL = sum_k_log_lambda - sum_lambda - sum_log_k_factorial
        
        # That was easy, right?  
        # Now, let's just convert from log-space to linear space and normalize 
        L = np.exp(LL)
        L /= np.max(np.max(L, axis=1, keepdims=True), axis=2, keepdims=True)
        
        # Actually that wasn't really the right normalisation for weighting,
        # although it was nice for plotting. We need weights to sum to 1.
        weights = L / np.sum(np.sum(L, axis=1, keepdims=True), axis=2, keepdims=True)
        if mode_resultant.lower() == 'mean':
            d1 = np.dot( np.sum(weights, axis=1), drift_vals)
            d0 = np.dot( np.sum(weights, axis=2), drift_vals)
        elif mode_resultant.lower() == 'peak':
            # Optimization notes:        
            # could cache these next two lines if you wanted.
            # would be less work to get peaks_mask without ge_iteriror condition applied,
            # then use only the peaks to index into the ge_interior to apply the condition.
            # the advantage being that you only have to do one reordering of the full sized array
            # rather than two...although maybe it doesn't matter much.
            
            # note that the weights>= interior is important, it ensures that
            # small local maximum are ignored if there are higher points equally 
            # close to the centre, but which are not maxima.
    
            h = np.hypot(drift_vals[:, _ax], drift_vals[_ax, :]) 
            h_sort_idx = np.argsort(h.ravel())
            weights_h_sorted = weights.reshape((len(weights), -1))[:, h_sort_idx]
            weights_sorted_ge_interior = weights_h_sorted >= np.maximum.accumulate(weights_h_sorted, axis=1)
            weights_ge_interior = np.empty(weights_h_sorted.shape, dtype=bool)       
            weights_ge_interior[:, h_sort_idx] = weights_sorted_ge_interior
            weights_ge_interior.shape = weights.shape
            peaks_mask = (grey_dilation(weights, size=(1, 3, 3) ) == weights) & weights_ge_interior
            # this next bit could be vectorized with soemthing like
            #  labels = np.repeat(np.arange(len(peaks_mask)), np.sum(peaks_mask, axis=0)) 
            #  and npg.aggregate_np(h[peaks_mask], labels, 'argmax')
            d0, d1, p = (np.full(len(peaks_mask), np.nan),
                         np.full(len(peaks_mask), np.nan),
                         np.full(len(peaks_mask), np.nan))
            for ii, peaks_mask_ii in enumerate(peaks_mask):
                if not np.any(peaks_mask_ii):
                    continue # leave nans for this iteration
                idx = np.argmin(h[peaks_mask_ii])
                idx_0, idx_1 =  peaks_mask_ii.nonzero()
                d0[ii] = drift_vals[idx_0[idx]]
                d1[ii] = drift_vals[idx_1[idx]]
                p[ii] = weights[ii, idx_0[idx], idx_1[idx]]
                            
        if return_extra:
            locals_ = locals()
            name_mapping = dict(likelihood='L', mle_height='p')
            return InfoRunDrift(**{k: locals_.get(name_mapping.get(k, k), None) 
                                   for k in InfoRunDrift._fields})
        else:            
            return d0, d1, p
            
    @append_docstring(get_mle_drift_info)
    def get_mle_drift_props(self, t, c, **kwargs):
        d0, d1, p = self.get_mle_drift_info(t, c, **kwargs)
        d = np.hypot(d0,d1)
        return MLEDriftProps(run_mean_drift=np.mean(d),
                             run_std_drift=np.std(d),
                             run_drift_likelihood_mean=np.mean(p),
                             run_drift_likelihood_std=np.std(p))
    
    @append_docstring(get_mle_drift_info)
    def plot_mle_drift_info(self, t, c, info=None, fn='generated\\out.mp4', **kwargs):
        """
        WARNING: clears the whole of the ``gcf``.
        """
        import matplotlib.animation as animation
        
        if info is None:
            info = self.get_mle_drift_info(t, c, return_extra=True, **kwargs)
        
        dpi = 100    
        fig = plt.gcf()
        fig.clf()
        fig.set_size_inches([5, 6])
        
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax.imshow(info.rm, cmap='jet', interpolation='nearest')
        
        frame = np.zeros_like(info.likelihood[0]) + np.nan
        im = ax.imshow(frame, cmap='jet', interpolation='nearest',
                       extent=[-info.S*2-2, -1, 2*info.S+1, 0])
        im.set_clim([0, 1])
        
    
        plt.tight_layout()
        
        plt.plot(-info.S-1, info.S, 'ws')
        pl = [None, None, None, None, None]
        def update_img(n):
            im.set_data(info.likelihood[n+1])
            xy = info.xy_bin_inds[:, info.pos_run_label==n+1]    
            if pl[0]:
                pl[0].pop(0).remove()
                pl[1].remove()
                pl[2].pop(0).remove()
                pl[3].pop(0).remove()
                pl[4].remove()
            pl[0] = plt.plot(xy[1], xy[0], 'w', lw=2, zorder=1)
            pl[1] = plt.scatter(*xy[::-1],
                                s=info.spike_counts[info.pos_run_label==n+1]*10,    
                                c='w', zorder=2, edgecolor='k')  
            pl[2] = plt.plot(info.d1[n+1]-info.S-1, +info.d0[n+1]+info.S, 'rs')
            ax.set_title('run #' + str(n+1))
            pl[3] = plt.plot(xy[1]+info.d1[n+1], xy[0]+info.d0[n+1], 'r', lw=2, zorder=1)
            pl[4] = plt.scatter(xy[1]+info.d1[n+1], xy[0]+info.d0[n+1],
                                s=info.spike_counts[info.pos_run_label==n+1]*10,
                                c='r', zorder=2, edgecolor='k')  

        ani = animation.FuncAnimation(fig, func=update_img,
                                      frames=len(info.likelihood)-1,
                                      interval=50, blit=False,
                                      init_func=lambda : None) 
        writer = animation.writers['ffmpeg'](fps=16)
        writer.bitrate=10000
        ani.save(fn, writer=writer, dpi=dpi)
        plt.clf()
        ax = plt.gca()
        plt.text(0.5, 0.5, ("saved run_mle_drift video as:\n{}\n\n[note: if you "
                            "have VLC media player you can use 'E' shortcut to step "
                            "through frames]").format(fn), ha='center', transform=ax.transAxes)
        plt.axis('off')
