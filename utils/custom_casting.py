# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 00:37:53 2014

@author: daniel
"""
import re
import numpy as np

num_re = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?") #copied from some suggestion somewhere in python docs as scanf substitute
def str2float(str):
    m = num_re.search(str)
    return float(m.group(0)) if m else None
    
def str2int(str):
    v = str2float(str)
    return int(v) if v is not None else None
    
def nanToEmptyStr(x=0,fmt="{}"):
    return '' if np.isnan(x) else fmt.format(x)

def NoneToNan(x):
    """None becomes nan, everything else passes uncahnged """
    return x if x is not None else np.nan
    
def str2html(s,nbsp=True):
    """Escapes &,<,>,",' characters and converts newlines to <br>
    if ``nbsp`` is     true, spaces are converted to non-breaking spaces
    """
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")\
         .replace("'", "&#39;").replace('"', "&quot;").replace("\n", "<BR>")
    if nbsp:
        s = s.replace(' ','&nbsp;')
    return s