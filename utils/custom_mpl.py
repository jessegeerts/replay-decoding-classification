# -*- coding: utf-8 -*-
"""
Module contents:

    # coolNumber
    # axes_equal, axes_square
    # trans_data_offset, trans_xdata_yaxes,trans_xaxes_ydata 
    # colorWithAplha, applyCmapAndAlpha
    # matshow_
    # imscatter, circscatter, textscatter
    # showAsPdf
    # HandlerLineCollectionMulti
    # significance_str, significance_legend, add_signficance_bar


@author: daniel

"""
import matplotlib.pylab as plt
from numpy import ma,nan,isnan,atleast_1d,empty,ones,array, floor, log10
from matplotlib.colors import Normalize as mplNormalize
from matplotlib.patches import Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as mplColormap
from matplotlib.collections import LineCollection
import matplotlib.transforms as transforms
from matplotlib.legend_handler import HandlerLine2D    
from matplotlib.lines import Line2D
from utils.custom_functools import append_docstring
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import hsv_to_rgb
import numpy as np
from io import BytesIO
from base64 import b64encode
import warnings

"""
When calling ax.set_aspect(float) I get strange warning saying that unicode
comparison failed.  I can't recreate this warning on the command line or pdb,
despite spending ages trying to.  I guess it must be some kind of bug in python?
"""
warnings.simplefilter("ignore", category=UnicodeWarning)

def coolNumber(x):
    """
    Takes a number x and rounds to the nearest "cool" number.
    see code for what cool means.
    """
    if 1 <= x <= 100:
        # return the next smallest value from the following list:
        possibles = array([1,2,4,5,10,15,20,25,40,50,60,75,100])
        p = possibles.searchsorted(x)
        return possibles[p] if x == possibles[p] else possibles[p-1]
    else:
        ret = 10**floor(log10(x))
        if x/ret >= 5:
            return ret*5
        elif x/ret >= 2.5:
            return ret*2.5
        else:
            return ret
            
def axes_equal(ax=None):
    """
    maintain existing xlim and ylim, but get x and y units represented equally.
    """
    ax = plt.gca() if ax is None else ax
    ax.set_aspect('equal', adjustable='box-forced')
        

def axes_y_eq_x(ax=None):
    """
    Set x and y lims to be the same, using min/max over x and y for choice.
    Also, show y=x line.
    """
    ax = plt.gca() if ax is None else ax
    
    xlim = sorted(ax.get_xlim())
    ylim = sorted(ax.get_ylim())
    
    lim = [min([xlim[0],ylim[0]]),max([xlim[1],ylim[1]])]
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal', adjustable='box-forced')
    ax.plot(lim,lim,'k:')
    
def axes_square(ax=None):
    """
    make shape of axes a square, ignoring units.
    """
    ax = plt.gca() if ax is None else ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xlim = sorted(xlim)
    ylim = sorted(ylim)
    ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]) )

    
def trans_data_offset(x=0,y=0,ax=None):
    """
    Use this in plot/scatter calls etc. as transform=trans_data_offset(y=10).
    This will use data coordiantes, but shift by 10 dots in the y direction.
    
    if `ax` is None, plt.gca() will be used.
    
    TODO: I can't remeber if this actually works properly.
    """
    ax = plt.gca() if ax is None else ax
    return transforms.offset_copy(ax.transData, fig=ax.figure, y=y,x=x, units='dots')
    
def trans_xdata_yaxes(ax=None):
    """
    Use this in plot/scatter calls etc. as transform=trans_xdata_yaxes().
    You can then use y values in axes coordiantes and x values in the usual data coordiantes.
    if `ax` is None, plt.gca() will be used.
    """
    ax = plt.gca() if ax is None else ax
    return transforms.blended_transform_factory(ax.transData, ax.transAxes)


def trans_xaxes_ydata(ax=None):
    """
    Use this in plot/scatter calls etc. as transform=trans_xaxes_ydata().
    You can then use x values in axes coordiantes and y values in the usual data coordiantes.
    if `ax` is None, plt.gca() will be used.
    """
    ax = plt.gca() if ax is None else ax
    return transforms.blended_transform_factory(ax.transAxes, ax.transData)
    

def colorWithAplha(color,alpha):
    """
    ``color`` is  one of the valid mpl color values e.g. 'r' or [1,0,0] or '#f00'
    ``alpha`` is a 1d array of length n, with values on the interval [0,1]
    
    The function returns an array of size nx4, repeating the specified color
    n times, and applying the specified alpha value in each case.
    
    """
    from matplotlib.colors import ColorConverter
    alpha = [alpha] if isinstance(alpha,(float,int)) else alpha
    ret = empty((len(alpha),4))
    ret[:,3] = alpha
    ret[:,:3] = ColorConverter().to_rgb(color)
    return ret
    
def applyCmapAndAlpha(Z,A,Zmax=None,Zmin=None,cmap=mplColormap.jet):
    """
    applies cmap to matrix Z, then sets alpha values according to matrix A.
    A should already be normalised to (0,1). Values outside this range will be clipepd to to (0,1).
    """
    if isinstance(Z, ma.MaskedArray):
        Z = Z.filled(nan)
        
    if isinstance(A, ma.MaskedArray):
        A = A.filled(0)
    A[isnan(Z)] = 0

    norm = mplNormalize(vmin=Zmin, vmax=Zmax)
    cmap = mplColormap.ScalarMappable(norm=norm, cmap=cmap)        
    cdata = cmap.to_rgba(Z)            
    cdata[...,3] = A.clip(0,1);
    
    return cdata
    
def matshow_vf(X,fignum=False,**kwargs):
    """
    """
    if fignum is False or fignum is 0:
        ax = plt.gca()
    else:
        # Extract actual aspect ratio of array and make appropriately sized figure
        fig = plt.figure(fignum, figsize=plt.figaspect(X))
        ax  = fig.add_axes([0.15, 0.09, 0.775, 0.775])

    mask = np.isnan(X)
    H = np.angle(X)/np.pi/2+0.5
    S = np.abs(X)

    if 'vmax' in kwargs:
        S = S.clip(0,kwargs['vmax'])
        S /= kwargs['vmax']
    else:
        S /= np.max(S[~mask])
        
    HSV = np.dstack((H,S,np.ones_like(S)))
    RGB = hsv_to_rgb(HSV)
    RGBA = np.dstack((RGB, ~mask))
    nr, nc = X.shape
    kw = {'origin': 'upper',
          'interpolation': 'nearest',
          'aspect': 'equal'}          # (already the imshow default)
    kw.update(kwargs)
    im = ax.imshow(RGBA, **kw)
    ax.title.set_y(1.05)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both')
    return im
    
def matshow_(Z,A,Zmax=None,Zmin=None,fignum=False,**kwargs):
    """
    like matplotlib's matshow, but takes a second paramter which is used for alpha values.
    Colormap here is fixed as jet..if you need to generalise then do so.
    
    A should be on the interval [0,1], we will clip values otuside this range
    """


    if fignum is False or fignum is 0:
        ax = plt.gca()
    else:
        # Extract actual aspect ratio of array and make appropriately sized figure
        fig = plt.figure(fignum, figsize=plt.figaspect(Z))
        ax  = fig.add_axes([0.15, 0.09, 0.775, 0.775])
        
            
    cdata =  applyCmapAndAlpha(Z,A,Zmax=Zmax,Zmin=Zmin,cmap=mplColormap.jet)       

    nr, nc = Z.shape
    kw = {'origin': 'upper',
          'interpolation': 'nearest',
          'aspect': 'equal'}          # (already the imshow default)
    kw.update(kwargs)
    im = ax.imshow(cdata, **kw)
    ax.title.set_y(1.05)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both')
    
    
    #ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
    #                                         steps=[1, 2, 5, 10],
    #                                         integer=True))
    #ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
    #                                         steps=[1, 2, 5, 10],
    #                                         integer=True))
    return im
    
def imscatter(x, y, im_array, zoom=1,vmin=None,vmax=None,xycoords='figure points',frameon=False,**kwargs):
    """
    Intended for scattering a list of images for ultimately rendering to pdf.
    
    ``x``,``y``, and ``im_array`` should be iterables of the same length.
    
    Each image can be a simple numpy array, or a 2-tuple of matching shaped arrays, where the second will be treated as alpha values.
    In either case, we convert masked arrays into nan-ified arrays, which will give us transparency at the nan bins.

    If the image is simply a scalar ``nan``, no image will be rendered at the corresponding (x,y).
    
    TODO: currently vmax and vmin only apply to 2-tuple alpha version.
    """
    
    ax = plt.gca()
    x, y = atleast_1d(x, y)
    artists = []
    for x0, y0,im in zip(x, y,im_array):
        if isinstance(im,float) and isnan(im):
            continue
        
        if isinstance(im,tuple):
            if len(im) == 2 and im[0].shape == im[1].shape:
                im = applyCmapAndAlpha(im[0],im[1],Zmin=vmin,Zmax=vmax)
                off_im = OffsetImage(im, zoom=zoom)
            else:
                raise Exception("imscatter called with a tuple that does not consist of a pair of matching arrays.")
        else:
            if isinstance(im, ma.MaskedArray):
                im = im.filled(nan)
              
            if vmax is not None or vmin is not None:
                # TODO: do we really have to do this here in order to deal with vmin and vmax?
                #      I think the None's dont work well when data has nan or something..need to check that too.
                norm = mplNormalize(vmin=vmin, vmax=vmax)
                cmap = mplColormap.ScalarMappable(norm=norm, cmap=mplColormap.jet)        
                cdata = cmap.to_rgba(im)          
                cdata[...,3] = ~isnan(im)
                im = cdata
            off_im = OffsetImage(im, zoom=zoom)
                
        ab = AnnotationBbox(off_im, (x0, y0), xycoords=xycoords, frameon=frameon,**kwargs)
        artists.append(ax.add_artist(ab))
    
    
def circscatter(x_vals,y_vals,scales,zorder,colors,vmin,vmax,cmap='RdYlBu',transform=None,clip_on=False,**kwargs):
    """
    Intended for scattering a list of filled cricles for ultimately rendering to pdf.    
     I couldn't make scatter work with transform=None, so lets do this instead
    
    ``x_vals``,``y_vals``,``scales`` and ``colors`` should be iterables of the same length.    
    
    At each x,y we render a circle of the specified scale and color.
    
    ``cmap`` is a string name of a colormap within matplotlib.cm.
    """
    fig = plt.gcf()
    cmap = mplColormap.ScalarMappable(norm=mplNormalize(vmin=vmin,vmax=vmax),cmap=getattr(mplColormap,cmap))    
    for x,y,s,c in zip(x_vals,y_vals,scales,colors):
        fig.artists.append(Circle((x,y),s,transform=transform,clip_on=clip_on,zorder=zorder,ec='k',fc=cmap.to_rgba(c),**kwargs))
    
def textscatter(x_vals,y_vals,txt_array,transform=None,clip_on=False,**kwargs):
    """Intended for scattering a list of strings for ultimately rendering to pdf.    
    ``x_vals``,``y_vals``,and ``txt_array`` should be iterables of the same length.    
    Elements in ``txt_array`` should be strings, which can include ``"\n"``newlines.
    Alternatively they can be a ``nan``, in which case nothing will be rendered at
    the corresponding (x,y) point.
    
    """
    for x,y,txt in zip(x_vals,y_vals,txt_array):
        if isinstance(txt,float) and isnan(txt):
            continue
        plt.text(x,y,txt,va='center',ha='center',transform=transform,clip_on=clip_on,**kwargs)
        
def plot_contingency_text(c,format_func='n={:g}'.format, ax=None,pad=0.02,**kwargs):
    """
    ``c`` is a 1d array which will be plotted as text in the corners of the 
    axes, using the following indexes-to-location mapping:
        
        1   |  3
        --------
        0   |  2
    """
    ax = plt.gca() if ax is None else ax
    plt.text(pad, pad, format_func(c[0]), va='bottom', ha='left',
             transform=ax.transAxes,**kwargs)
    plt.text(pad, 1-pad, format_func(c[1]), va='top', ha='left',
             transform=ax.transAxes,**kwargs)
    plt.text(1-pad, pad, format_func(c[2]), va='bottom', ha='right',
             transform=ax.transAxes,**kwargs)
    plt.text(1-pad, 1-pad, format_func(c[3]), va='top', ha='right',
             transform=ax.transAxes,**kwargs)

    
    
def showAsPdf(name,figs=None,closefigs=True,showTitle=True,showPageNumbers=True,PDF_FOLDER=""):
    """
    This will save with the given name in the current directory, appending 
    a one letter suffix if the file is currently open.  Otherwise it will
    create or overwrite the file with the given name.
    """
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    import subprocess
    
    if figs is None:
        figs = plt.gcf()
    if not isinstance(figs,(list,tuple)):
        figs = [figs]
        
    try:
        if showTitle or showPageNumbers:
            for f_i, f in enumerate(figs):
                f.text(0.99,0.01,
                    (name if showTitle else '') + \
                    ("" if len(figs) == 1 else " p" + str(f_i+1) + " of " + str(len(figs)) ) ,\
                    va='bottom',ha='right')
            
    except Exception:
        pass
    
    fn_base = os.getcwd() + os.sep + PDF_FOLDER + os.sep + name
    for suffix in ['','_a','_b','_c','_d','_e','_f','_g']:
        try:
            fn = fn_base + suffix + ".pdf" 
            with PdfPages(fn) as pdf:
                for f in figs:
                    pdf.savefig(f,papertype='a4')
            print("Saved figure as : %s" % fn)
            if closefigs:
                for f in figs:
                    plt.close(f)
            break # if we've got this far then we've done the save
        except IOError as e:
            if e.errno != 13:
                raise 
    subprocess.Popen([fn],shell=True)
    
    #os.system('"%s"' % (fn)) # This methods seems to cause matplotlib to crash
    

class HandlerLineCollectionMulti(HandlerLine2D):
    
    def get_numpoints(self, legend):
        if self._numpoints is None:
            return legend.scatterpoints
        else:
            return self._numpoints
  
    def _default_update_prop(self, legend_handle, orig_handle):
        read_val = lambda  x,i: x[i % len(x)]
        lw = read_val(orig_handle.get_linewidth(),self._line_ii)
        dashes = read_val(orig_handle.get_dashes(),self._line_ii)
        color = read_val(orig_handle.get_colors(),self._line_ii )
        legend_handle.set_color(color)
        legend_handle.set_linewidth(lw)
        if dashes[0] is not None: # dashed line
            legend_handle.set_dashes(dashes[1])
  

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
  
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = ((height-ydescent)/2.)*ones(xdata.shape, float)
        
        n_lines = len(orig_handle.get_paths())
        arts = []
        extent = 0.6*height
        for ii in xrange(n_lines):
            legline = Line2D(xdata, ydata - extent/2. + (n_lines-ii)/n_lines*extent)
            self._line_ii = ii
            self.update_prop(legline, orig_handle, legend)
            legline.set_transform(trans)
            arts += [legline]
            
        return arts

""" use as ``handler_map`` arg in legend...or combine with other key-vals to make more complex map``"""
handler_map_linecollectionmulti = {LineCollection: HandlerLineCollectionMulti()}


def significance_str(p=0,n=None,p_thresh=[0.05,0.01],
                     not_signficant_str='n.s.',label=False,
                     for_markdown=False):
    """
    The output is a string, e.g.::
    
        n=23
         **
         
    If ``n`` is ``None``, no n value will be shown.
    
    One star will be shown for each of the ``p_thresh`` values crossed by ``p``.
    (``p`` must be smaller than the values in ``p_thresh``).
    If no thresholds are crossed then ``not_signficant_str`` will be shown.
    If ``not_signficant_str`` is ``None`` the function will return ``''``, even if
    an ``n`` value was provided.  You can provide ``not_signficant_str=''`` if
    you want to show n, but not give any not signficant string.
    
    If ``label`` is ``True`` then we put ``p < 0.05`` or whatever the relevant
    threshold is.
    """
    p_thresh = sorted(p_thresh,reverse=True)
    n_thresh = sum(p<array(p_thresh))
    
    if n_thresh > 0:
        ret =  (("n=%d" % n) if n is not None else '') + \
                    (', p<{:g}\n'.format(p_thresh[n_thresh-1]) if label \
                        else '\n') + '*' * n_thresh
    elif not_signficant_str is not None:
        ret = (("n=%d\n" % n) if n is not None else '') + not_signficant_str
    else:
        ret = ''
        
    if for_markdown:
        return ret.replace('\n',' ').replace('*','\\*')
    else:
        return ret
        
def significance_legend(p=[],p_thresh=[0.05,0.01,0.001],
                        not_signficant_str='n.s.',ax=None,
                        pattern="Signficance:\n{ signficance }",
                        loc=(0.05,0.05)):
    """
    This produces a legend for string made with the ``significance_str`` function.
    
    ``p`` is an array of p values.
    ``p_thresh`` has the same meanding as ing ``significance_str``.
    ``not_signficant_str`` should be the same value that you passed to ``significance_str``.
    
    The output string shows only those entries that are relevant, eg::
    
        n.s. not signficiant
        *    p < 0.05
        ***  p < 0.001
                
    """
    relevant = array([False]*(len(p_thresh)+1))
    for pp in p:
        n_thresh = sum(pp<array(p_thresh))
        relevant[n_thresh] = True
    
    if not any(relevant):
        return ''
        
    if not_signficant_str == '' or not_signficant_str is None:
        relevant[0] = False
        
    w = array([0 if not_signficant_str is None else len(not_signficant_str)] + range(1,len(p_thresh)+1))
    w = w[relevant]
    if len(w) == 0:
        return ''
        
    w = max(w)

    s = [("{:<" + str(w) +"} not significant").format(not_signficant_str)] + \
            [("{:<" + str(w) +"} p < {:g}").format("*"*(ii+1),pp) for ii, pp in enumerate(p_thresh)]
    
    if len(s) == 0:
        return
        
    s =  '\n'.join([ss for ss,r in zip(s,relevant) if r])
    ax = ax if ax is not None else plt.gca()
    ax.annotate(pattern.replace('{ signficance }',s),loc,xycoords='axes fraction',size=16)
        
@append_docstring(significance_str)
def add_signficance_str(x,y,p=0,n=None,p_thresh=[0.05,0.01],
                        not_signficant_str='n.s.',ax=None):
    """
    Adds significance text above a point See ``significance_str`` for details
    of the string arguments.

    You can provide an array of length ``m``, for: ``x``, ``y``, ``p``, and ``n``.
    or a scalar.
    """
    vars_with_len = sum(hasattr(g,'__len__') for g in (x,y,p,n))
    if vars_with_len == 0:
        x,y,p,n = (x,),(y,),(p,),(n,)
    elif vars_with_len == 4:
        m = len(x)
        if any(len(g) != m for g in (x,y,p,n)):
            raise Exception("x,y,p,n must be of the same length.")
    else:
        raise Exception("x,y,p,n must either all be scalars or all be arrays.")
                            
    ax = ax if ax is not None else plt.gca()

    for x_,y_,p_,n_ in zip(x,y,p,n):
        s_ = significance_str(p=p_, n=n_, p_thresh=p_thresh,
                                not_signficant_str=not_signficant_str)        
        if len(s_) == 0:
            continue
        ax.annotate(s_, (x_,y_), xytext=(0, 15), va='bottom', ha='center',
                textcoords='offset points')
            
            
            
def add_signficance_bar(start, end, p=0,n=None,p_thresh=[0.05,0.01],not_signficant_str='n.s',
                        txt_kwargs=None, arrow_kwargs=None,ax=None,label=True,**kwargs):
    """
    Adds a bar with asterisks and/or n. The bar spans from x_start to x_end, at height 
    given by max(y_start,y_end).

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to draw to

    start : (x,y)
        
    end : (x,y)

    The signficnace string is created using ``significance_str``.  The relevant args are:
    ``p``, ``p_thresh``, ``n``, ``not_signficant_str``. ...see ``significance_str``
    help for details.
    

    txt_kwargs : dict or None
        Extra kwargs to pass to the text

    arrow_kwargs : dict or None
        Extra kwargs to pass to the annotate
        

    partly adapted from:
    Taken from: http://stackoverflow.com/a/18386751/2399799
    """

    ax = plt.gca() if ax is None else ax
    
    txt_str = significance_str(p=p,n=n,p_thresh=p_thresh,
                               not_signficant_str=not_signficant_str,
                               label=label)
    
    if txt_kwargs is None:
        txt_kwargs = {}
    if arrow_kwargs is None:
        arrow_kwargs = {'arrowprops':dict(arrowstyle="-",
                            connectionstyle="arc3",
                            ec="k",
                            )}

    trans = ax.get_xaxis_transform()

    y_max = max(start[1],end[1])
    ann = ax.annotate('', xy=(start[0],y_max),
                        xytext=(end[0],y_max),
                        transform=trans,
                        **arrow_kwargs)
    txt = ax.annotate(txt_str,
                      ((start[0] + end[0]) / 2.0,y_max),
                      ha='center',
                      va='bottom',
                      **txt_kwargs)


    if plt.isinteractive():
        plt.draw()
    return ann, txt
    

def inset_hist(data, level_x, level_stack, ax=None,
                    width="30%", height="20%", loc=3, ax_off=(0.05,0.55,1,1),
                    stack_color_dict=None):
    """
    This does a fairly specific thing.
    
    ``data`` is a dataframe with at least two levels in it's multiindex.
    One of these levels must be chosen as the x-axis, and the other will be
    used for grouping/stacking.
    
    Uses the first column's counts, i.e. if there are different numbers of nans
    in the multiple columns you may not be getting what you expect.
    """
    main_ax = plt.gca() if ax is None else ax
    
    data_hist = data.groupby(level=[level_x,level_stack])\
                                         .count()[data.columns[0]]\
                                         .reset_index()\
                                         .pivot(columns=level_stack,
                                                index=level_x)
    
    ax = inset_axes(main_ax, width, height, loc=loc, bbox_to_anchor=ax_off,
                    borderpad=0, bbox_transform=main_ax.transAxes)

    # reverse columns so that the "first" column appears at the top of the bar
    if stack_color_dict is not None:
        colors = [stack_color_dict[c] for _, c in data_hist.columns[::-1]]
    else:
        colors = None        
    data_hist[data_hist.columns[::-1]] \
        .plot(ax=ax, kind='bar', stacked=True, legend=False, sharex=False,
                   color=colors)
    
    ax.set_axis_bgcolor('none')
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.tick_params(axis='x',top='off')
    ax.grid(False)
    ax.set_yticks([])    
    xlocs, xlabels = plt.xticks()
    ax.set_xticks([xlocs[0], xlocs[-1]])
    ax.set_xticklabels([xlabels[0].get_text(),xlabels[-1].get_text()])
    
    plt.setp(xlabels, rotation=0)
    plt.xlabel(level_x)
    ax.xaxis.labelpad = -15
    plt.axes(main_ax)
    return ax    
    
    
def inset_bar_graph(data_list, x_labels=None, yerr=None, y_lims=None, ax=None,
                    width="20%", height="25%", loc=3, ax_off=(0.05,0.55,1,1),
                    inset=True, color = [0.4,0.4,0.4]):
    """
    If ``x_labels`` is none it uses A-Z.
    ``data_list`` is a squence of y values, ``yerr`` is a matching sequence
    of y error values.
    If ``ax`` is none it uses ``gca``. If ``y_lims`` is None
    it copies the x_lims from the parent ``ax``.
    
    returns a reference to the inner axes
    
    The bars are centred on x = 0,1,2,...
    
    If ``inset`` is false then we don't actually create the inset axes, but use
    ``ax`` or ``gca``.
    
    You can insert ``None`` in the list of y values, in order to add horizontal
    spaces.  The x_labels and yerr should have matching ``None``s.
    """
    
    main_ax = plt.gca() if ax is None else ax
    if inset:
        ax = inset_axes(main_ax, width, height, loc=loc, bbox_to_anchor=ax_off,
                        borderpad=0, bbox_transform=main_ax.transAxes)
    else:
        ax = main_ax
        
    ax.set_axis_bgcolor('none')
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('k')
    ax.spines['left'].set_color('none')
    
    x_vals = np.arange(len(data_list))
    if x_labels is None:
        x_labels = tuple('ABCDEFGHIJKLNOPQRSTUVWXYZ'[:len(x_vals)])
    if yerr is None:
        x_vals, x_labels, data_list = zip(*[(x, y) for x, xl, y in
                                        zip(x_vals, x_labels, data_list)
                                        if y is not None])
    else:
        x_vals, x_labels, data_list, yerr = zip(*[(x, xl, y, err) for x, xl, y, err in
                                        zip(x_vals, x_labels, data_list, yerr)
                                        if y is not None])
        
    ax.bar(x_vals, data_list, yerr=yerr, align='center',
           color=color, error_kw=dict(elinewidth=3, ecolor='k'))
    ax.set_xticks(x_vals)
    ax.set_xlim([-0.6, x_vals[-1]+0.6])

    ax.set_xticklabels(x_labels)
    if y_lims is None:
        y_lims = main_ax.get_xlim()
    ax.set_ylim(y_lims)
    ax.set_yticks(y_lims)
    ax.yaxis.tick_right()
    ax.tick_params(axis='x',top='off')
    ax.grid(False)
    #ax.tick_params(axis='x', colors='none')
    #ax.tick_params(axis='y', colors='none')
    plt.axes(main_ax)
    return ax
    
def figure_to_html(fig=None,width=None,height=None,border_css='',hover_html=None):
    """
    Returns the figure (or gcf) as a base64 encoded html image
    ``hover_html`` can be some html, which will be added as a hover element on
    the image.  If you do this, you need to include the ``figure_to_html.hover_header_el``
    string in the head section of your html.
    """
    fig = plt.gcf() if fig is None else fig
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')    
    img = '<img style="border:' + border_css + \
            '" src="data:image/png;base64,' + b64encode(buf.getvalue()).decode() + '" />'
    if hover_html:
        img = '<div class="img_hoverable" style="display:inline-block;position:relative;">' + img \
              + '<div class="img_hover" style="background:rgba(255,200,200,0.6);padding:3px;position:absolute;display:none;top:0px;left:0px;">' + hover_html + \
              "</div></div>"
    return img
figure_to_html.hover_header_el = "<style>.img_hoverable:hover .img_hover{display:block !important;}</style>"
    
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)
        
