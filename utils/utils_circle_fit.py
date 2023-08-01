# -*- coding: utf-8 -*-

import numpy as np

def fit_circle(x,y):
    """
	http://www.scipy.org/Cookbook/Least_Squares_Circle

	Given two arrays of x and y coordinates it returns
		(x_center, y_centre, radius)
	of the circle fitted in the least squares sense.  
	
	The fitting is done by linearising the problem.
    """
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m
    
    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    
    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)
    
    xc = x_m + uc
    yc = y_m + vc
    
    # Calculation of all distances from the center (xc_1, yc_1)
    Ri      = np.sqrt((x-xc)**2 + (y-yc)**2)
    R       = np.mean(Ri)
    
    return xc, yc, R
