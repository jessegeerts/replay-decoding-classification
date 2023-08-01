# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection as mpl_LineCollection
import matplotlib.cm as mpl_colormap
import seaborn as sns

from utils.custom_mpl import coolNumber as cool_number, make_segments, matshow_vf
from utils.custom_functools import append_docstring


from core.trial_basic_analysis import TrialBasicAnalysis as AnalysisClass # used only for docstrings

PALETTE_ARRAY =[
    [0.8627, 0.8627, 0.8627],
    [0, 0, 0.7843],
    [0.3137, 1, 0.3137],
    [1, 0, 0],
    [0.9608, 0, 1],
    [0.2941, 0.7843, 1],
    [0, 0.7255, 0],
    [1, 0.7255, 0.1961],
    [0, 0.5882, 0.6863],
    [0.5882, 0, 0.6863],
    [0.6667, 0.6667, 0],
    [0.7843, 0, 0],
    [1, 1, 0],
    [0.5490, 0.5490, 0.5490],
    [0, 1, 0.9216],
    [1, 0, 0.6275],
    [0.6863, 0.2941, 0.2941],
    [1, 0.6078, 0.6863],
    [0.7451, 0.7451, 0.6275],
    [0.8824, 0.8824, 0.2941]]


class TrialBasicPlotting():
    """This is a mixin for plotting stuff generated in trial_basic_analysis.py"""
    
    @append_docstring(AnalysisClass.get_spa_ratemap)
    def plot_spa_ratemap(self, t=None, c=None, made_earlier=None, vmax=None,
                         *args, **kwargs):
        """see trial.get_ratemap for explanation of arguments
            alternativelty, pass in a custom ``made_earlier`` ratemap for plotting.
        """
        if made_earlier is not None:
            rm = made_earlier
        else:
            rm, _ = self.get_spa_ratemap(t, c, *args, **kwargs)

        plt.cla()
        gca = plt.gca()
        if self.pos_shape is not None:
            pos_clip = self.pos_shape.clip_region(lw=2, ec='k')
            gca.add_patch(pos_clip)
        else:
            pos_clip = None

        plot_func = plt.imshow if not np.iscomplexobj(made_earlier) else matshow_vf
        im = plot_func(rm, interpolation="nearest",
                   extent=[0, self.w, self.h, 0], vmax=vmax)
        if pos_clip is not None:
            im.set_clip_path(pos_clip)
        
        # Plot scale bar    
        scale_len = cool_number(self.w*0.2) 
        plt.plot([self.w*0.75, self.w*0.75 + scale_len],
                  [0.95*self.h, 0.95*self.h], linewidth=3, color=[0, 0, 0])
        gca.text(self.w*0.75+scale_len*0.9, 0.95*self.h,'%dcm' % (scale_len),
                 ha='right', va='bottom')
        
        plt.axis('scaled')
        plt.axis([-0.5, self.w-0.5, -0.5, self.h-0.5]) #imshow puts the centre of each pixel at integer coordinates
        gca.invert_yaxis()
        gca.patch.set_visible(False)
        gca.set_xticks([])
        gca.set_yticks([])
        if vmax is None:
            gca.title.set_text("Max={:0.0f}{}".format(np.amax(rm), "")) # TODO: find out units from get_spa_rm

    @append_docstring(AnalysisClass.get_dir_ratemap)
    def plot_dir_ratemap(self, t=None, c=None, ax=None, **kwargs):
        dir_rm, bin_lower_end = self.get_dir_ratemap(t, c, **kwargs)
        if ax is None:
            ax = plt.axes(polar=True)
        ax.plot(np.append(bin_lower_end/180.*np.pi, bin_lower_end[0]),
                 np.append(dir_rm, dir_rm[0]))
        
    @append_docstring(AnalysisClass.get_theta_hist)
    def plot_theta_hist(self, t, c=None, color=None, cla=True, 
                        show_info_text=True, **kwargs):
        h, bin_edges, nan_count, total_count = self.get_theta_hist(
                                        t=t, c=c, return_extra=True, **kwargs)        
        gca = plt.gca()        
        if cla:       
            plt.cla()
        if color is None:
            color = PALETTE_ARRAY[c] if c is not None else [0.8]*3
        plt.step(x=bin_edges, y=np.append(h, h[-1]), where='post', 
                 lw=2, color=color)
        gca.set_ylim(0, 2.5/len(bin_edges))
        gca.set_xlim(0, 360)
        gca.set_yticks([])
        gca.set_xticks([0, 180, 360])
        plt.xlabel('theta phase')
        if show_info_text:
            plt.text(0.98,0.98,"phase unavailable for\n{} of {} spikes".format(
                                                        nan_count, total_count),
                    va='top',ha='right', transform=gca.transAxes)
        
    @append_docstring(AnalysisClass.get_t_ac)
    def plot_t_ac(self, t=None, c=0, color=None, cla=True, **kwargs):
        """plots the temporal autocorr."""
        gca = plt.gca()        
        if cla:
            plt.cla()
            ylim_old = 0
        else:
            ylim_old = gca.get_ylim()[1]
        h, bin_edges, mean_rate = self.get_t_ac(t, c, return_extra=True, **kwargs)
        plt.step(x=bin_edges, y=np.append(h, h[-1]), where='post', lw=2,
                 color=PALETTE_ARRAY[c] if color is None else color)
        
        gca.set_xlim(0, bin_edges[-1])
        gca.set_ylim(0, max(ylim_old, np.max(h[1:])*1.2))
        if mean_rate is not None:
            gca.text(0.02, 0.95, "{0:.2}Hz".format(mean_rate), ha='left', va='top',
                     transform=gca.transAxes)
        gca.set_yticks([])
        gca.set_xticks([0, bin_edges[int(len(bin_edges)/2)], bin_edges[-1]])
        plt.xlabel('time (ms)')
        plt.draw_if_interactive()
        
            
    @append_docstring(AnalysisClass.get_speed_hist)
    def plot_speed_hist(self, **kwargs):
        h, bin_lefts, bin_widths = self.get_speed_hist(return_extra=True, **kwargs)
        plt.cla()
        plt.bar(bin_lefts[1:-1], h[1:-1], width=bin_widths[1:-1])
        plt.xlim(bin_lefts[0], bin_lefts[-1] + bin_widths[-1])
        small_cm_per_s = min(bin_widths[1]/2, bin_widths[0]/2)
        if h[0] > h[1]:
            plt.plot(bin_lefts[[1, 1]], h[[1, 0]], 'k:')
        plt.plot([bin_lefts[1], bin_lefts[1] - small_cm_per_s],
                 h[[0, 0]], 'k:')
        if h[-1] > h[-2]:
            plt.plot(bin_lefts[[-1, -1]], h[[-2, -1]], 'k:')
        plt.plot([bin_lefts[-1], bin_lefts[-1] + small_cm_per_s],
                 h[[-1, -1]], 'k:')

        gca = plt.gca()                
        gca.set_xticks(bin_lefts[1:])
        gca.set_yticks([])
        plt.xlabel('speed (cm/s)')
        plt.draw()

    def plot_shape(self, w=None, h=None, bin_size_cm=2.5, **kwargs):
        """simply plots the pos shape"""
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        self.pos_shape.plot(w=w, h=h, bin_size_cm=bin_size_cm, **kwargs)
        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.draw()
            
    def plot_pos(self, t=None, c=None, colors=None, colors_vmax=4, 
                 show_shape=False, xy=None, w=None, h=None):
        """
        Shows the pos path. Optionally plotting spikes for one or more cells.
        To specify one cell do ``t=2,c=3`` or whatever, to specify multiple cells
        on the same tetrode do ``t=2,c=(1,2,3)`` or whatver, to specify multiple cells
        across tetrodes do ``t=(1,3,4,4),c=(1,1,1,2)`` or whatever. Do do all cells on a tet
        do ``t=2`` or whatever.
        
        If you only plot cells on a single tetrode the color used for each cell is
        the standard tint colors.  If you plot cells form multiple cells then the order
        of the cells in the list determines the colors, using the tint colors starting from 1.
    
        You can alternatively pass in a colors vector and colorsvmax. This only makes
        sense when plotting a single cell. If you provide these arguments each spike
        will be plotted in the specified color, using scatter rather than plot.
        
        ``xy``, ``w``, and ``h``  are normally ``None`` meaning use ``trial.xy``
        etc., but you can override that here.
        """
        
        plt.cla()
        gca = plt.gca()        
        # gca.hold(True)  # not Python 3 compatible
        plt.axis('equal')
        gca.patch.set_visible(False)
        
        if xy is None:
            xy = self.xy
        if h is None:
            h = self.h
        if w is None:
            w = self.w
            
        plt.plot(xy[0], xy[1], c='k', alpha=0.4, rasterized=True)
            
        if t is not None or self.recording_type == 'npx':
            # see info at top of function for possible meanings of t and c
            if c is None:
                c = self._available_cells(t)
                t = [t]*len(c)
            else:
                if not isinstance(c, (list, tuple)):
                    c = [c]
                if not isinstance(t, (list, tuple)):
                    t = [t]*len(c) 
                                  
            if colors is None:
                if len(np.unique(t)) > 1:
                    plot_colors = [PALETTE_ARRAY[ii+1] for ii in range(len(c))]
                else:
                    plot_colors = [PALETTE_ARRAY[cc % len(PALETTE_ARRAY)] for cc in c]
            else:
                scatter_colors = colors
                
            for ii, (t_ii, c_ii) in enumerate(zip(t, c)):
                pos_idx = self.spk_times(t=t_ii, c=c_ii, as_type='p')
                pos_xy = xy[:,pos_idx]
                if colors is None:
                    plt.plot(pos_xy[0], pos_xy[1], '.', c=plot_colors[ii], 
                             rasterized=True)            
                else:
                    plt.scatter(pos_xy[0], pos_xy[1], c=scatter_colors,
                                alpha=0.8, edgecolors='none',
                                s=25, vmax=colors_vmax, rasterized=True)          
                                
        if show_shape:
            self.pos_shape.plot()
            
        # Plot scale bar    
        scaleLen = cool_number(w*0.2)
        plt.plot([w*0.75, w*0.75 + scaleLen],
                 [0.95*h, 0.95*h], linewidth=3, color=[0, 0, 0])
        gca.text(w*0.75+scaleLen*0.9, 0.95*h, '%dcm' % (scaleLen),
                 ha='right', va='bottom')
            
        #gca.hold(False)
        gca.set(adjustable='box') #plt.axis('scaled')
        plt.axis([0, w, 0, h])
        gca.invert_yaxis()
        gca.set_xticks([])
        gca.set_yticks([])
        #plt.show()
        
    @append_docstring(AnalysisClass.get_t_rate)
    def plot_pos_alt(self, t=None, c=None, colors=None, vmin=0, vmax=None,
                     title=None, xy=None, w=None, h=None, show_shape=True):
        """
        This is very similar to plot_pos, but here each segment of the path is colored
        rather than having a gray path with colored spikes.
        
        If ``t`` and ``c`` are supplied then this will plot the temporarly smoothed
        rate along the path...no it wont.
        
        If ``colors`` is supplied it should be an array of the same length
        as ``self.xy`` gving either a single value for each item in xy or an rgba value.
        If a single value is provided (i.e. 1d vector rather then jet colormap will be applied)
        
        Note that this offers fewer options than ``plotPos``, mainly because it doesn't
        make sense to try and plot multiple things here.
        """
        if colors is None:
            rate = self.get_t_rate(t=t, c=c)
            if title is None:
                max_rate = np.amax(rate)
                if vmax is None:
                    vmax = max_rate
                title = "max={:0.1}Hz".format(max_rate)                 
            colors = rate # colormap is applied next...
    
        if colors.ndim == 1:
            if vmax is None:
                vmax = np.max(colors)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = mpl_colormap.ScalarMappable(norm=norm, cmap=mpl_colormap.jet)
            colors = cmap.to_rgba(colors)
            
        xy = self.xy if xy is None else xy
        w = self.w if w is None else w
        h = self.h if h is None else h
        plt.cla()
        gca = plt.gca()        

        lc = mpl_LineCollection(segments=make_segments(xy[0], xy[1]),
                                color=colors, linewidth=2, rasterized=True)
        gca.add_collection(lc)
        
         # Plot scale bar    
        scaleLen = cool_number(w*0.2)
        plt.plot([w*0.75, w*0.75 + scaleLen],
                 [0.95*h, 0.95*h], linewidth=3, color=[0, 0, 0])
        gca.text(w*0.75+scaleLen*0.9, 0.95*h, '%dcm' % (scaleLen),
                 ha='right', va='bottom')
            
        if show_shape and self.pos_shape is not None:
            self.pos_shape.plot()
        plt.axis('scaled')
        plt.axis([0, w, 0, h])
        gca.invert_yaxis()
        gca.set_xticks([])
        gca.set_yticks([])
        
        if title is not None:
            plt.title(title)
        plt.show()  
        
    def plot_theta(self, t=None, c=None):
        """Basically t and c don't matter, but we offer them here to maintain
        convention. In fact we do something vaguely usful: we show spikes on
        the plot in the right temporal locations.
        """
        times = np.arange(len(self.eeg()))/ float(self.eeg_samp_rate)
        s = np.nanmax(self.eeg())        
        yax2=25
        
        plt.plot(times, self.eeg(bad_as=None) * (10./s) + yax2, c=[0.8]*3)
        plt.plot(times, self.eeg() * (10./s) + yax2, 'k')
        plt.plot(times, self.filtered_eeg() * (10./s) + yax2, 'r')        
        plt.plot(times, self.theta_amp* (10./s) + yax2, 'm',lw=2)
        plt.plot(times, self.theta_phase, 'b')
        plt.plot(times, self.theta_freq, 'c')
        if t is not None:
            spk_times = self.spk_times(t, c, as_type='s')
            plt.plot(spk_times, np.full(len(spk_times), 15), 'sy', mew=0)
        plt.legend(('raw eeg', 'good eeg', 'filtered eeg', 'theta amplitude','theta phase (radians)',
                    'theta freq (Hz)'))
        plt.xlabel('time (seconds)')
        plt.xlim(0, self.duration)
        
    def plot_eeg_spectrogram(self, win_width_s=2, max_freq_hz=60):
        plt.specgram(self.eeg(bad_as=None), NFFT=int(self.eeg_samp_rate*win_width_s),
                     Fs=self.eeg_samp_rate)
        plt.xlabel('time (s)')
        plt.ylabel('freq (Hz)')
        plt.ylim(0,max_freq_hz)