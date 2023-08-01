# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:39:14 2015

@author: daniel
"""

class PointsOfInterestNotFound(Exception): #great name for an exception class, eh?
    pass

class DontTrustThisAnalysisException(NotImplementedError):
    pass
