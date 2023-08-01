# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 17:50:26 2015

@author: daniel
"""

#import pandas as pd


import numpy as np
from io import BytesIO
from PIL import Image
from base64 import b64encode
from matplotlib import cm
from matplotlib.colors import Normalize, hsv_to_rgb
from matplotlib.colors import Colormap as ColormapClass
from custom_mpl import applyCmapAndAlpha
   
def pd_matshow(vmin=None,
               vmax=None,
               cmap=cm.jet,
               zoom=None,
               multi='min'):
    """
    cmap can be something from ``matplotlib.cm`` or, if ``x`` is iterable, it can 
    be a tuple/list specifying the colormap for each item in ``x``, e.g. 
    ``cmap=(cm.Reds,cm.Greens)``.  The ``multi`` arg specifies how to combine these
    multiple images, currently accepts``"or","min"``, ``"max"`` or an integer index
    into the list of x. It can also be ``"alpha"``, in which ``x`` should be a
    sequence of two matricies, the second is the alpha values to apply to the first.
    
    Alternatively, ``cmap`` can be a comma-delimited string with some combiantion of
    the letters RGBA: e.g. ``R,B``. This will use the first index of ``x`` for the red
    channel, the second for the blue channel etc.
    Another alternative: ``cmap``, can be ``"complex"``, in which case we render using 
    HSV, with -pi to pi being H, and 0 to max(abs(x)) being S. ``vmin`` and ``vmax`` are
    ignored for now.
    """
    def matshow_func(x):
        is_2darray = lambda x: isinstance(x, np.ndarray) and x.ndim == 2
        c = None
        
        if isinstance(multi,int):
            x = x[multi]  
        elif isinstance(multi, basestring) and multi.lower() == "alpha":
            #if not is_2darray(x[0]) or is_2darray(x[1]) or x[0].shape != x[1].shape:
            #    return ""
            c = applyCmapAndAlpha(x[0],x[1],Zmin=vmin,Zmax=vmax,cmap=cmap)
            c = (c*255).astype(np.uint8)
            
        if c is None and not is_2darray(x):
            return ""

        b = BytesIO()  
        norm = Normalize(vmin,vmax,clip=True)    
        
        if c is not None:
            pass # we already made c, above
        elif isinstance(cmap,ColormapClass):
            if not np.ma.isMaskedArray(x):
                x = np.ma.array(x,mask = np.isnan(x))  
            c = cmap(norm(x), bytes=True)
        elif isinstance(cmap,basestring):
            if cmap.lower() == 'complex':
                if isinstance(x,np.ma.MaskedArray):
                    x = x.filled(np.nan)
                mask = np.isnan(x)        
                H = np.angle(x)/np.pi/2+0.5
                S = np.abs(x)
                if vmax is not None:
                    S = S.clip(0, vmax)
                    S /= vmax
                else:
                    S /= np.max(S[~mask])

                HSV = np.dstack((H,S,np.ones_like(S)))
                c = np.zeros(shape=x.shape + (4,),dtype=np.uint8)
                c[...,3] = (~mask)*255
                c[...,:3] = hsv_to_rgb(HSV)*255
            else:
                c = np.zeros(shape=x[0].shape + (4,),dtype=np.uint8)
                c[...,3] = 255
                for color, xx in zip(cmap.split(','),x):
                    plane = 'RGBA'.index(color)
                    c[...,plane] = norm(xx)*255
                if not np.ma.isMaskedArray(c):
                    mask = reduce(np.logical_or,[np.isnan(xx) for xx in x])
                    c[...,3] = (~mask)*255
        elif isinstance(cmap,(tuple,list)):
            c = []
            for mp, xx in zip(cmap,x):
                c.append(mp(norm(xx),bytes=True))
            if multi.lower() == 'or':
                c = reduce(np.bitwise_or,c)
            elif multi.lower() == 'min':
                c = reduce(np.minimum,c)
            elif multi.lower() == 'max':
                c = reduce(np.maximum,c)
            else:
                raise NotImplementedError()
                
            if not np.ma.isMaskedArray(c):
                c = np.ma.array(c,mask = np.isnan(c))  
        else:
            raise Exception()
        

            
        img = Image.fromarray(c)
        if zoom is not None:
            img = img.resize(tuple(int(s*zoom) for s in img.size),Image.ANTIALIAS)
        img.save(b, format='png')
        return '<img alt="2d array" src="data:image/png;base64,' + b64encode(b.getvalue()) + '" />'
        
    matshow_func.escape = False  # This prevents the "<" tags getting escaped
    matshow_func.justify = 'all' # This prevents the long string of data getting abrieviated with "..."
    return matshow_func
    

def pd_circshow(vmin=0,vmax=5,cmap=cm.jet_r):
    """
    TODO: really we probably want to have pixels and vmin and vmax.
    """
    norm = Normalize(vmin, vmax, clip=True)
    
    def circshow_func(v):
        if np.isnan(v):
            return ""
        x = norm(v)
        
        k = np.arange(-vmax,+vmax+1)**2
        m = k[:,np.newaxis] + k[np.newaxis,:] < (x*vmax)**2
        c = np.zeros((vmax*2+1,vmax*2+1,4),dtype=np.uint8)
        c[m] = cmap(x,bytes=True)
        b = BytesIO()  
        Image.fromarray(c).save(b, format='png')       
        return "{:g}".format(v) + ' <img alt="2d array" src="data:image/png;base64,' + b64encode(b.getvalue()) + '" />'
        
    circshow_func.escape = False  # This prevents the "<" tags getting escaped
    circshow_func.justify = 'all' # This prevents the long string of data getting abrieviated with "..."
    return circshow_func


def pd_boolshow(true_str='{ tick }',false_str='{ cross }'):
    """
    Icons from: https://www.iconfinder.com/icons/22536/.
    
    """
    def boolshow_func(x):
        if x:
            return true_str.replace('{ tick }', ("<img src='data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAACOElEQVQ4T2NkoADUr1MWYyRXf8NaFRlGRsbjZBnQsFZNiYnx/4m///"
            "5zk2xAzQpVTVZ2xkP//v4X+ffnXyRJBtSuUTFkYWbcw872l+/rV8bNzWF3g4g2oHaViiUT8/9tYiJ/BV68YH7568N/9c70ex/BBtQv0"
            "bNgZmQ5zsL3yarK985x9ICtWi7nzMLCslZO9i//4wdMn779+u/XHffgIEgdI0xzXHA0w/Jd8779Y/zuV+17by/MkJrVcj7//zItUVH+"
            "z//6JcPXd+8Y53bEPciHyTM2LTX6r2f6h+Hrt88MhkrRDKt2L//I8I/BqyH07rHa1QoR/34zzDQ14eN7+erj/zu3Ge49ePhQa3UDwy+"
            "4AVlTRXnYWThvGJkyCzEz/+MUFVBmOHDk3gd2Fs51P35/DzEz4+P78vULw8Vzfz/9+P3fcnLGo2vIXgSHQUG/gAATD98lQxM2cWbmP2x"
            "cHHwMly/8ZtA1YGX4/uMzw43r/758/sjQ2Jf2sAc9fOCxkNYjJcLDy3LJyIxDlIn5F0uk7n2G5ZcUGN6+Zvn74N6fM71pDy0ZGBj+4z"
            "QAJJE1VVSCk4XropYRhwgn+0+mf/84GM6f+v7h65e/OjOKnjzFluwx0kHWVClZThaW87rGvEK3b3z5+u3z/+T+zEercOUZbAlJUteW18"
            "wlWnDp01vfq1b1vQali38MDAw/GRgYvjEwMHxnYGB4Do8FLCZLMzAwsKuacvMx/P398/a5X28ZGMDR9gGbKwA40dqSw4SMsQAAAABJRU"
            "5ErkJggg=='/>"))
        else:
            return false_str.replace('{ cross }',("<img src='data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAACO0lEQVQ4T41SzWsTQRx9k91kN19LWkNMrPZYECrRgoKoB0XFgh/YBb/Ol"
            "voXeJVe+w/4dbd4GBUUKij2oObQSykVCnqzpmmNjW1S87HZzMjM7CQhB3EuOzvv997vzfsNQd/6ejM6ZZqE+j53x543XvwPRnSRJh+6N4"
            "f1R/fRL/IvTAqIAsMgNHtuEq0vnxE9eRnFV8/AGHcFHjIIHbl2B/XCG9hHTqH0YQGso1wSQQYJ0ZHzl9BY+QgQAnAgfvoK1l/Oiy1Gr9/G"
            "n0+vA7Mc0fwZFN+/BThzydqNOM9N5OF9/wYIP4IhFgGGbj2Q29/zs/JY2g1q7LGjKBYKkA68jkH3pS1F5EFNUCgNibPgqwW2t5uIGMztZl"
            "BvGzQ9HFZdBpY2pQ2UKz5iYV9l0D+FWtOg6SEDRNpQWQwqlisMSVuRtaDUeHc2NeUkGM2kdL/BvkqtvAPs7hnuhcWdnoAgx2OMZpxOr6O8e"
            "F+o2ioBtnZN1OshKUIE2bIYzTpedwB9tTpXPRjwQHWraqHVIi5ZvJjih9M1lCod2PFIdxLNuofcsCH/JRaLdB1pbO1XAtIBMUDHM0okbJto"
            "N31JXv2ZBOHA+H6FRSwTXstH2DKxuZcAZ1BjFCKMgOazqlCQVzYdhLh6ygyg+VwPE2erpaTKQN9XiLQ5oRMHqlguOTDBu0kLzAehx3JVLG84"
            "MEkPEwJZAKbI+/GJ5OTBqPl0o+HfnV6qicfvBXOwHx5PXh2NmU9+NPzpmaXaQnDu/wVrzRtU8J7mzAAAAABJRU5ErkJggg==' />"))
    boolshow_func.escape = False  # This prevents the "<" tags getting escaped
    boolshow_func.justify = 'all' # This prevents the long string of data getting abrieviated with "..."
    return boolshow_func
    