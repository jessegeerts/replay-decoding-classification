# -*- coding: utf-8 -*-
"""

Based on Yoon, Fiete, Barry et al. 2013. but not identical.

By DM, oct 2014.
        
"""
from numpy import sqrt, matrix, cos, sin, array, abs, minimum, linspace, pi, tensordot, argmin, Inf, unravel_index, \
    arange, \
    ceil, meshgrid, vstack, floor, random, ones, sum

from time import time

ratio = 2 / sqrt(
    3)  # this makes a rectangle of the right proportions, to give equilateral triangles when alternate rows are half-shifted


def fit_peaks(p, w, orient_n=30, scales=linspace(6, 15, 40), phase_nish=20, print_time_info=False):
    """
    Uses dist_to_peak, trying multiple orientations and scales.
    
    ``orient_n`` - specify the number of orients to try on [0,60] degrees    
    ``scales`` - list of scales in the same units as p   
    ``phase_nish`` - specify the number of phases in the "x" axis.  In the y axis    
                    we use sqrt(3)/2 * phase_nish ... hence the "ish".
                    Set to None to use only phase (0,0), i.e. peak at origin.
                    
    `p` is the [nx2] array of peaks to fit to
    `w` is the [n] array of peak weights
    
    
    """

    start_time = time()

    orients = linspace(0, pi / 3, orient_n, endpoint=False)

    # Note that this isn't a perfect grid, in terms of equal and uniform x and y steps but it's basicalyl ok
    if phase_nish is None:
        v = (array([0]), array([0]))
    else:
        v = (linspace(0, 1, num=phase_nish, endpoint=False),
             linspace(0, 1, num=phase_nish / ratio, endpoint=False))

    best_C = Inf
    best_orient = best_scale = best_phase = None
    for orient in orients:
        p_rot = rotate(p, orient)
        for scale in scales:

            # compute dist to peaks for all phase offsets at this scale and orientation
            d = dist_to_peak_prerotated(p_rot, scale, v)

            # multiply distances by weights and sum over all peaks to get score for each phase offset
            C = tensordot(d, w, axes=([0], [0]))

            # find minimum score and potentially update the `best_` values
            min_C_ind_0, min_C_ind_1 = unravel_index(argmin(C, axis=None), C.shape)  # if phase_nish > 1 else 0,0
            min_C = C[min_C_ind_0, min_C_ind_1]

            if min_C < best_C:
                best_C = min_C
                best_orient = orient
                best_scale = scale
                best_phase = (v[0][min_C_ind_1], v[1][min_C_ind_0])

    if print_time_info:
        print("fitting took {:0.2f} seconds [peaks_n={}, orient_n={}, scales_n={}, phase_total_n={}]".format(
            time() - start_time,
            len(p),
            orient_n,
            len(scales),
            len(v[0]) * len(v[1])
        ))

    return best_C, best_orient, best_scale, best_phase


def rotate(p, orient):
    mat = matrix([[cos(-orient), sin(-orient) * ratio],
                  [-sin(-orient), cos(-orient) * ratio]])
    return array(matrix(p) * mat)


def dist_to_peak(p, orient=0, scale=10, v=[0, 0]):
    """
    This function wraps ``rotate`` and ``dist_to_peak_prerotated``.  Note that
    the ``fit_peaks`` function calls the two sub functions directly rather than
    using this function, this means it can avoid rotating multiple times for 
    all the scales at a given orientation.
    
    orient should be in radians, scale in same units as p.
    v should be [x,y] in [0,1]x[0,1]
    
    returns the sqr distance to the nearest peak for all points p, where p is shape nx2.    
    """
    return dist_to_peak_prerotated(rotate(p, orient), scale=scale, v=v)


def dist_to_peak_prerotated(p2, scale=10, v=[0, 0]):
    """
    orient should be in radians, scale in same units as p.
    v should be [x,y] in [0,1]x[0,1]
    
    returns the sqr distance to the nearest peak for all points p, where p is shape nx2.
    
    """
    scale = float(scale)

    # rotate and scale x and y    
    p2 = p2 * (1.0 / scale)  # note we need to take a copy of p2

    # shift every other "y-band" by 0.5 in x
    alternateRowsMask = (floor(p2[:, 1]).astype(int) & 1).astype(bool)  # floor(y mod 2) == 1
    p2[alternateRowsMask, 0] += 0.5

    # collapse everything down to [0,1]x[0,1]
    p2 %= 1

    # the nearest "version" of v, is either in the same "y-band" as p, or it's in one of the "y-bands" either side
    # we consider the two cases separately, but the two calcualtions share the same starting point of dx and dy
    # Note that once we know the distance in y axis we need to scale it by `ratio` to get its unit to agree with the x axis.

    if not isinstance(v, tuple):
        # this is the easy version if v is a single point
        dy = abs(p2[:, 1] - v[1])
        dx = abs(p2[:, 0] - v[0])
    else:
        # here v is a tuple of (x values,y values) which together define a grid
        dy = abs(p2[:, 1].reshape(-1, 1, 1) - v[1].reshape(1, -1, 1))
        dx = abs(p2[:, 0].reshape(-1, 1, 1) - v[0].reshape(1, 1, -1))

    # convert dx from phase to absolute units, and get dx_2 (the dx for the either-side y-bands)
    mask = dx > 0.5
    dx *= scale
    dx_2 = dx - 0.5 * scale  # (dx-0.5) *scale
    dx_2[~mask] *= -1  # we want to xor the sign bit of dx_2 with ~mask
    dx[mask] = scale - dx[mask]

    # do the same for y, computing the total distance for the same y-band and then for the either-side y-bands
    dy *= scale / ratio
    d1 = dx ** 2 + dy ** 2
    dy = scale / ratio - dy  # (1-dy) * scale/ratio
    d2 = dx_2 ** 2 + dy ** 2
    d = minimum(d1, d2)

    return d


def make_grid(orient=pi / 3, scale=10, v=[0, 0], xlim=[-20, 20], ylim=[-20, 20]):
    """ 
    Returns an [nx2] array giving the coordinates of peaks within the
    bounding box described by ``xlim`` and ``ylim``.
    
    ``orient`` gives the orienation in radians about (0,0)
    ``scale`` gives the grid scale in the same units as xlim and ylim
    ``v`` - is a 2-element list/tuple giving the phase in [0,1] x [0,1].
            Remember that the second dimension is not the same scale as the first.
            
    Note that this function is not used within the dist_to_peak function:
    there we avoid actually making a grid.  This funcion is needed for
    generating test data and for plotting results.
    
    TODO: Note that if the window is a long way from the origin then the grid
    may not cover all the window...could sort this out but I haven't.
    """

    # we do everything centered on the origin and then at the end translate
    # to the actual xlim,ylim
    m = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    m = ceil(m / scale) * 2
    cx, cy = round((xlim[1] - xlim[0]) / 2 / scale), round((ylim[1] - ylim[0]) / 2 / scale)
    x, y = meshgrid(arange(cx - m, cx + m), arange(cy - m, cy + m))

    x[(y.astype(int) & 1).astype(bool)] += 0.5

    x += v[0]
    y += v[1]
    mat = matrix([[cos(orient), sin(orient)],
                  [-sin(orient) / ratio, cos(orient) / ratio]]) * scale

    p = array(matrix(vstack((x.ravel(), y.ravel()))).T * mat)

    # p[:,0] += 0.5 * (xlim[1]+xlim[0])
    # p[:,1] += 0.5 * (ylim[1]+ylim[0])

    p = p[(p[:, 0] > xlim[0]) & (p[:, 0] < xlim[1]) & (p[:, 1] > ylim[0]) & (p[:, 1] < ylim[1])]

    return p
    # plt.clf(), plt.plot(p[:,0],p[:,1],'.r'), plt.gca().set_aspect('equal', adjustable='box'), plt.xlim(*xlim),plt.ylim(*ylim),plt.draw()


def test_a(orient=pi / 3, scale=10, v=[0, 0], xlim=[-20, 20], ylim=[-20, 20]):
    """ create 40,000 uniformly random points and runs dist_to_peak
        using the values sepcified in the args here.
        
        The 40,000 are plotted with scatter, showing the distance as the colors.
        The requested grid is overlayed as red dots.
    """
    import matplotlib.pylab as plt

    p = random.rand(40000, 2)
    p[:, 1] *= ylim[1] - ylim[0]
    p[:, 1] += ylim[0]
    p[:, 0] *= xlim[1] - xlim[0]
    p[:, 0] += xlim[0]

    d = dist_to_peak(p, orient, scale, v=v)
    plt.clf()
    plt.scatter(p[:, 0], p[:, 1], 10, c=d, lw=0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(*xlim);
    plt.ylim(*ylim);

    p = make_grid(orient, scale, v=v, xlim=xlim, ylim=ylim)
    plt.scatter(p[:, 0], p[:, 1], 15, 'r')


def test_b(orient=pi / 3, scale=10, v=[0, 0], xlim=[-20, 20], ylim=[-20, 20], noise=0, plot=True, **kwargs):
    """ 
    For the given orient, scale, phase (``v``) and xlim and ylim, a grid is produce.
    if`noise` is none-zero unifrom jitter on [-noise,+noise] is added in x and y dimension to 
    grid.
    
    The ``fit_peaks`` function is then called with the generated grid.  ``**kwargs`` are
    passed to the ``fit_peaks`` function.
    
    If ``plot=True`` the ground truth data and fit data as plotted.
    
    The output of fit_peaks is returned and the expected distance value (given the noise jitter)
    
    """

    p = make_grid(orient, scale, v=v, xlim=xlim, ylim=ylim)
    p_noise = random.rand(*p.shape) * noise * 2 - noise

    p += p_noise
    w = ones(len(p)) if noise == 0 else noise ** 2 - sum(p_noise ** 2,
                                                         axis=1)  # make the most jittered points have the lowest weight

    best_C, best_orient, best_scale, best_phase = fit_peaks(p, w, **kwargs)

    expected_C = sum(sum(p_noise ** 2, axis=1) * w)  # compute what C should be if the ground truth fit is best
    # note that if noise is large enough and/or search space doesn't include
    # the true grid parameters then you can't actually expect this C
    if plot:
        import matplotlib.pylab as plt
        p2 = make_grid(best_orient, best_scale, v=best_phase, xlim=xlim, ylim=ylim)
        plt.clf()
        plt.plot(p[:, 0], p[:, 1], '.r')
        plt.plot(p2[:, 0], p2[:, 1], '.k')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(*xlim);
        plt.ylim(*ylim);

        plt.title(
            "(truth,fit): C=({:0.3f},{:0.3f}) \n ({:0.2f},{:0.2f}) units | ({:0.1f},{:0.1f}) degrees | (({:0.2f},{:0.2f}),({:0.2f},{:0.2f}))".format(
                expected_C,
                best_C,
                scale, best_scale, orient / pi * 180, best_orient / pi * 180, v[0], v[1], best_phase[0],
                best_phase[1]));

    return best_C, best_orient, best_scale, best_phase, expected_C


def test_c(orients=linspace(0, pi, 360, endpoint=False),
           scales=linspace(5, 20, 30),
           vs=(linspace(0, 1, num=25, endpoint=False),
               linspace(0, 1, num=int(25 / ratio), endpoint=False)),
           xlim=[-20, 20], ylim=[-20, 20], noise=0, **kwargs):
    """
    calls test_b with a range of scales,orients,and phases.
    **kwargs are passed through to test_b and on to find_peaks.
    
    TODO: actually make this usefully do something...currently it would take
    a stupid amount of time to finish..2 months maybe...and it doesn't even
    record the outputs or explore jitter space!
    """

    raise NotImplementedError("cant really explore 'all' of parameter space, need a smaller test size")
    # shape = (len(scales),len(orients),len(vs[0]),len(vs[1]))
    # best_C = empty(shape)
    # best_C[:] = nan

    for ii, scale in enumerate(scales):
        print("scale ", ii + 1, " of ", len(scales))
        for jj, orient in enumerate(orients):
            print("orient ", jj + 1, " of ", len(orients))
            for v_0 in vs[0]:
                for v_1 in vs[1]:
                    best_C, best_orient, best_scale, best_phase, expected_C = \
                        test_b(orient=orient, scale=scale, v=[v_0, v_1], xlim=xlim, ylim=ylim, noise=noise, plot=False,
                               **kwargs)
                    print((best_C, best_orient, best_scale, best_phase))
