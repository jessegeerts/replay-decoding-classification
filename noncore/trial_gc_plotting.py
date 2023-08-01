# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.custom_functools import append_docstring
from utils.custom_mpl import (matshow_, axes_square, matshow_vf, coolNumber as cool_number)
import matplotlib.cm as mpl_colormap
from noncore import dist_to_peak
from noncore import run_peak_rates

from noncore.trial_gc import TrialGCAnalysis as AnalysisClass # used only for docstrings


class TrialGCAnalysisPlotting(object):
    """mixin allied to trial_gc.py"""
    
    @append_docstring(AnalysisClass.get_spa_ac_peak_fit)
    def plot_spa_ac_peak_fit(self, t=None, c=None, *args, **kwargs):
        F = self.get_spa_ac_peak_fit(t=t, c=c, return_extra=True, **kwargs)
        
        g = dist_to_peak.make_grid(F.best_orient, F.best_scale, F.best_phase,
                                   xlim=[-F.center[0], F.center[0]],
                                   ylim=[-F.center[1], F.center[1]]) + np.array(F.center)
        
        plt.cla()
        matshow_(F.ac, F.n, Zmin=-1, Zmax=1, fignum=False) #special version of matshow that takes alpha values matrix as second arg
        plt.plot(F.coords[1], F.coords[0], 'xk', mew=2, alpha=0.3)
        plt.plot(g[:, 1], g[:, 0], 'ok', mew=2, mfc='none')
        plt.xlim(0, F.n.shape[1]); plt.ylim(0, F.n.shape[0]);
        plt.title("scale={:0.1f} cm".format(F.best_scale*F.bin_size_cm))
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    @append_docstring(AnalysisClass.get_spa_ac)
    def plot_spa_ac(self, t=None, c=None, show_n=False, *args, **kwargs):
        """
        Can supply ac for plotting or provide inputs to pass on to get_spa_ac.
        
        Here we have one extra argument:
        ``show_N`` - if true, we ask the getAutocorr function to return the
        number of bins in the correlation and normalise it. We then use this
        value as the alpha channel of the image.  Note that you shouldn't
        try and pass ``return_n`` and ``normalise_n`` here - use this special
        ``shown_N`` arg instead.

        """
        if 'ac' in kwargs:
            ac = kwargs['ac']
            bin_size_cm = kwargs.get('bin_size_cm', None)
        else:
            ac, bin_size_cm = self.get_spa_ac(t, c, return_n=show_n, 
                                              normalise_n=True, *args, **kwargs)
 
        if bin_size_cm is None:
            bin_size_cm = 1
            bin_units = 'units'
        else:
            bin_units = 'cm'
            
        if show_n:
            matshow_(ac[0], ac[1], fignum=False, Zmin=kwargs.get('vmin', -1),
                     Zmax=kwargs.get('vmax', 1))
            ac = ac[0]
        else:
            plt.matshow(ac, fignum=False, vmin=kwargs.get('vmin', -1),
                        vmax=kwargs.get('vmax', 1))
        gca = plt.gca()

        # Plot scale bar    
        wB = ac.shape[1]*bin_size_cm # we get the cool number in cm, then convert back to bins
        scaleLenB = cool_number(wB*0.2) 
        scale_len = scaleLenB/bin_size_cm
        w, h = ac.shape
        plt.plot([w*0.75, w*0.75 + scale_len], [0.95*h, 0.95*h], linewidth=3, 
                 color=[0, 0, 0])
        gca.text(w*0.75+scale_len*0.9, 0.95*h, '%d%s' % (scaleLenB, bin_units),
                 ha='right', va='bottom')
        
        gca.set_xticks([])
        gca.set_yticks([])
        plt.xlim(0, h)
        plt.ylim(w, 0)

    def plot_pos_with_gc_field_measures(self, t, c, colors=None):
        self.plot_pos(t=t, c=c, colors=colors)
        gca = plt.gca()
        info =  self.get_gc_field_measures(t, c)
        plt.imshow(info.boundary_labels[::-1, :],
                   extent=[0, self.w, 0, self.h], 
                   interpolation='none', zorder=10)
        ax = plt.gca()
        for c in info.circs:
             ax.add_patch(plt.Circle((c.xc, c.yc), radius=c.r, zorder=1, fill=False, lw=2))
        
    @append_docstring(AnalysisClass.get_gc_fields)
    def plot_pos_with_gc_fields(self, t, c, colors=None, **kwargs):
        """ Calls .plot_pos and then adds on top the result of .get_gc_fields
        """
        self.plot_pos(t=t, c=c, colors=colors)
        gca = plt.gca()
        field_info = self.get_gc_fields(t, c, as_boundaries=True, as_masked=True,
                                        return_extra=True, **kwargs)
        plt.imshow(field_info.labels[::-1, :], extent=[0 , self.w, 0, self.h], 
                                            zorder=0, interpolation='none')
        if field_info.peak_bin_inds is not None:     
            plt.plot(field_info.peak_bin_inds[1][1:]*field_info.bin_size_cm,
                     field_info.peak_bin_inds[0][1:]*field_info.bin_size_cm, 'ok')
        
#        if self._posShape is not None:
#            self._posShape.plot(self.w, self.h, binSize)
    
    def plot_ratemap_with_gc_field_measures(self, t, c, get_circs=True, **kwargs):
        """
        Note that the ratemap displayed is just a default ratemap, i.e. it doesnt
        use the settings chosen for actually getting the fields.
        """
        self.plot_spa_ratemap(t=t, c=c)
        gca = plt.gca()
        info =  self.get_gc_field_measures(t, c, get_circs=get_circs, **kwargs)
        ax = plt.gca()
        if get_circs:
            for c in info.circs:
                 ax.add_patch(plt.Circle((c.xc, c.yc), radius=c.r, zorder=1, fill=False, lw=2))
        plt.plot(info.peak_xy[1][1:], info.peak_xy[0][1:], 'ok')
        if info.tri is not None:
            plt.triplot(info.peak_xy[1][1:], info.peak_xy[0][1:], info.tri.simplices.copy()[info.tri_used], lw=3, color='w')
            plt.triplot(info.peak_xy[1][1:], info.peak_xy[0][1:], info.tri.simplices.copy(), lw=1, color='k')
            plt.title('scale={:0.1f}cm'.format(info.tri_scale))        
                
    @append_docstring(AnalysisClass.get_spa_stability)
    def plot_spa_stability(self, t, c, **kwargs):
        """Uses getSpaStability to get a ratemap a pair of ratemaps for the first
        and second halves of the trial.  It then the first ratemap into the red
        channel, and the second into the green channel. White bins are where 
        at least one of the ratemaps had no dwell.
        Note we use adaptive smoothing because it seems to give better coverage
        for the cut-down size of the data.
        """
        S = self.get_spa_stability(t, c, return_extra=True, **kwargs)
        rm = np.dstack((S.a/np.amax(S.a), S.b/np.amax(S.b), np.zeros_like(S.a)))
        rm[S.a.mask | S.b.mask] = 1
        self.plot_spa_ratemap(made_earlier=rm)
        plt.title("R={:0.2f} | p<{:0.2f}".format(S.r, S.p+0.01))
        

    @append_docstring(AnalysisClass.get_gravity_transform)        
    def plot_gravity_transform(self, t, c, sum_mode='complex', show_shape=True, **kwargs):
        G = self.get_gravity_transform(t, c, sum_mode=sum_mode,
                                       nodwell_mode='ma', **kwargs)
        
        if sum_mode=='simple':                                     
            self.plot_spa_ratemap(made_earlier=G)
            plt.colorbar()
        elif sum_mode == 'complex':
            matshow_vf(G, interpolation="nearest", extent=[0, self.w, self.h, 0])
        else:
            raise Exception("what?")
        plt.title('Gravity transform')
        
        if show_shape is True and self.pos_shape is not None:
            self.pos_shape.plot()
            
        
    @append_docstring(run_peak_rates.plot_pos_peak_info)        
    def plot_pos_peak_info(self, t=None, c=None, **kwargs):
        return run_peak_rates.plot_pos_peak_info(self, t, c, **kwargs)
        
    @append_docstring(AnalysisClass.get_rate_ll)
    def plot_rate_ll(self, t, c, **kwargs):
        G = self.get_rate_ll(t=t, c=c, return_extra=True, **kwargs)
        self.plot_pos_alt(colors=G.pos_likelihood, vmin=0, vmax=1)
        plt.title('LL={:g}'.format(G.LL_total))
        # TODO: show colorbar from 0 to 1

    @append_docstring(AnalysisClass.dist_to_peak)
    def plot_dist_to_peak(self,t, c, shape_kwargs={}, dist=None, cx=None, 
                          cy=None, **kwargs):
        if None in (dist, cx, cy):
            if dist is not None or cx is not None or cy is not None:
                raise Exception("provide evrything or nothign.")                
            rm, dist, cx, cy = self.dist_to_peak(t, c, **kwargs)
            
        spk_idx = self.tet_times(t, c, as_type='p')        
        plt.cla()
        plt.scatter(self.xy[0, spk_idx], self.xy[1, spk_idx], s=20, c=dist, lw=0)
        plt.xlim(0, self.w)
        plt.ylim(0, self.h)
        axes_square()
        plt.gca().invert_yaxis()
        plt.plot(cy, cx, 'xw', mew=3), plt.plot(cy, cx, 'xk', mew=1)
        if self.pos_shape is not None:
            self.pos_shape.plot(**shape_kwargs)

    @append_docstring(AnalysisClass.get_spa_ac_windowed)
    def plot_spa_ac_windowed(self, t, c, **kwargs):
        ac, bin_size_cm = self.get_spa_ac_windowed(t=t, c=c, return_extra=True,
                                                   **kwargs)
        if kwargs.get('as_1d', False) is True:
             plt.plot(np.arange(len(ac))*bin_size_cm, ac)  
             plt.xlabel('distance from spike (cm)')
        else:
            self.plot_spa_ac(ac=ac, bin_size_cm=bin_size_cm, vmin=None, vmax=None)
                       
    def plot_theta_hist_in_out_gc_field(self, t, c):
        """
        Histogram of theta phases, restricted to spikes in the field when ``in_out_field=True``
        and restricted to spikes out of the field when ``in_out_field=False``.
        """
        spk_pos_ind = self.tet_times(t=t, c=c, as_type='p')        
        pos_is_in_field = self.get_pos_is_in_field(t, c)
        mask = pos_is_in_field[spk_pos_ind]
        
        self.plot_theta_hist(t, c, spike_mask=mask, color='r', show_info_text=False)
        self.plot_theta_hist(t, c, spike_mask=~mask, color='b', show_info_text=False,
                             cla=False)
        plt.legend(['in field', 'out of field'])
        
    def plot_pos_t_ac(self, t, c, window_ms=[50., 150.]):
        """
        Scatter the spikes on top of the path and color the spikes by the
        number of other spikes located in a pair of windows either side
        of the given spike, e.g. the previous and next theta cycles but not
        the current theta cycle.
        """
        times = self.tet_times(t, c)
        #times.sort() # times should darn well be sorted already 
        
        n_adjacent_spikes = np.zeros(len(times), dtype=int)
        window_s = np.array(window_ms)/1000.
        
        for i, tt in enumerate(times):
            istart, iend = times.searchsorted(tt + window_s, side='right')
            n_adjacent_spikes[i] += iend-istart
            n_adjacent_spikes[istart:iend] += 1

        self.plot_pos_with_gc_fields(t, c, n_adjacent_spikes)
                    
    def plot_t_ac_in_out_gc_field(self, t, c, **kwargs):
        """
        Plots a standard-ish temporal autocorr twice, once using only spikes in
        the field and once using only spikes outside the field. 
        Note that the get_* sister function takes an extra flag to control 
        which of the two verions you want, whereas here we show both.
        
        kwargs passed through to plot_t_ac
        """
        spk_pos_ind = self.tet_times(t=t, c=c, as_type='p')
        pos_is_in_field = self.get_pos_is_in_field(t, c)
        mask = pos_is_in_field[spk_pos_ind]
        
        self.plot_t_ac(t, c, spike_mask=mask, color='r', **kwargs)
        self.plot_t_ac(t, c, spike_mask=~mask, color='b', cla=False, **kwargs)
        plt.legend(['in field', 'out of field'])                
        plt.text(0.02, 0.98, "overal rate: {0:.2}Hz".format(len(spk_pos_ind)/self.duration),
                 ha='left', va='top', transform=plt.gca().transAxes)
                    
    @append_docstring(AnalysisClass.get_gc_measures)
    def plot_gc_measures(self, t=None, c=None, made_earlier=None, **kwargs):
        """Passes args through to get_gc_measures."""
        
        if made_earlier is None:
            info = self.get_gc_measures(t=t, c=c, **kwargs)
        else:
            info = made_earlier
            
        ax = plt.gca()
        ax.imshow(info.ac, cmap=mpl_colormap.gray_r, interpolation='nearest', 
                  vmin=-1, vmax=1)
        #ax.hold(True)
        ax.imshow(info.ac_masked, cmap=mpl_colormap.viridis, interpolation='nearest',
                  vmin=-1, vmax=1) #TODO: need to put it in the right place
        line_width = 2

        # horizontal green/white dashed line at 3 o'clock
        pXY = info.closest_peaks_coord
        ax.plot((pXY[0, 1], info.ac.shape[1]), (pXY[0, 0], pXY[0, 0]), '-g',
                lw=line_width, zorder=8)
        ax.plot((pXY[0, 1], info.ac.shape[1]), (pXY[0, 0], pXY[0, 0]), 'w', 
                lw=line_width, ls='dashed', zorder=8)
        
        # Plot the arc showing the orientation
        theta1, theta2 = (-info.orientation, 0) if info.orientation > 0 else (0, -info.orientation)
        mag = info.scale / info.bin_size_cm * 2 # x2 is because we need diameter of Circle not radius
        a = patches.Arc(pXY[0, ::-1], mag, mag,
             theta1=theta1, theta2=theta2, linewidth=line_width, fill=False, 
             zorder=5, color='r')
        ax.add_patch(a)
        a = patches.Arc(pXY[0, ::-1], mag, mag,
             theta1=theta1, theta2=theta2, linewidth=line_width, fill=False,
             zorder=5, color='w', ls='dashed')
        ax.add_patch(a)
                
        # plot lines from centre to peaks above middle
        for i in np.arange(1, pXY.shape[0]):
            ax.plot((pXY[0, 1], pXY[i, 1]), (pXY[0, 0], pXY[i, 0]), 'k',
                    lw=line_width)
            ax.plot((pXY[0, 1], pXY[i, 1]), (pXY[0, 0], pXY[i, 0]), 'w',
                    lw=line_width, ls='dashed')   
            
        all_ax = ax.axes
        x_ax = all_ax.get_xaxis()
        x_ax.set_tick_params(which='both', bottom='off', labelbottom='off', 
                             top='off')
        y_ax = all_ax.get_yaxis()
        y_ax.set_tick_params(which='both', left='off', labelleft='off',
                             right='off')
        all_ax.set_aspect('equal')
        all_ax.set_xlim((0.5, info.ac.shape[1]-1.5))
        all_ax.set_ylim((info.ac.shape[0]-.5, -.5))
        ax.title.set_text("{:0.0f}cm | G={:0.2f}{:}".format(info.scale,
                          info.gridness, ' (E)' if info.used_ellipse else ''))
        plt.draw()
        
    @append_docstring(AnalysisClass.get_gridness_and_shuffle)
    def plot_gridness_and_shuffle(self, t, c):
        plt.gca()
        info, shuffles = self.get_gridness_and_shuffle(t, c)
        self.plot_gc_measures(made_earlier=info)
        divider = make_axes_locatable(plt.gca())
        pannelB = divider.append_axes("bottom", size=1.2, pad=0.1)
        pannelB.plot(shuffles, np.arange(len(shuffles)), color='k')
        pannelB.plot([info.gridness]*2, [0, len(shuffles)], color='r')
        pannelB.text(info.gridness, 0 ,  "{:1.2f}".format(info.gridness), ha='right', va='bottom')
        plt.xticks([-1, 0, 1])
        plt.yticks([0, len(shuffles)])
        plt.draw()
        
