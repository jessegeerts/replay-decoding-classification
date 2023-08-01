# -*- coding: utf-8 -*-
"""
Adapted from the Matlab code acompanying the following paper:

Climer, J. R., DiTullio, R., Newman, E. L., Hasselmo, M. E., Eden, U. T.
(2014), Examination of rhythmicity of extracellularly recorded neurons 
in the entorhinal cortex. Hippocampus, Epub ahead of print. 
doi: 10.1002/hipo.22383.

Matlab code take from: github.com/jrclimer/mle_rhythmicity, on 02-feb-2015.

This python version is by DM, Feb 2015.

Module contents:

    # namedtuple ClimerPHat
    # namedtuple ClimerFit
    # statpoisci        <-- port of a Matlab stats function
    # rhythmicity_pdf
    # plot_rhythmicity_pdf   <-- calls rhythmicity_pdf and then produces a simple plot
    # theta_mle_climer  <-- this is the main entry point for mle-fitting
    # plot_theta_mle_climer  <-- this is a wrapper around theta_mle_climer, it plots outputs

In the comments, the abreviation [S.O.] means valid only when theta_skipping=True

Uses modfied version of pyswarm code for partical swarm optimsation. See:
    https://gist.github.com/d1manson/40cbbb62a5f4ecc37bd7
    or original at http://pythonhosted.org/pyswarm/
    
Note that original Matlab code used MLE after PSO, looking at Matlab it seems that
MLE is the Nelder-Mead algorithm, here we use L-BFGS-B instead. You can selecct
Nelder-Mead in scipy.optimise.minimze, but it doesnt let you specify bounds,
which is annoying.


Only limited testing has been done. Use at your own risk, but please let the 
author know if you find bugs!
"""

from collections import namedtuple
from numpy import sqrt
from numpy import pi, cos, exp
from scipy.stats import chi2, norm
import numpy as np
from utils.utils_pso import pso  # Partical swarm optimisation
from scipy.optimize import minimize
import matplotlib.pylab as plt
from utils.custom_np import repeat_ind, count_to
from utils.custom_functools import append_docstring

realmin = np.finfo(np.double).tiny  # same as in matlab

ClimerPHat = namedtuple('ClimerPHat', ', '.join((
    'tau',  # Exponential falloff rate of the whole distribution (log10(sec))
    'b',  # baseline probability
    'c',  # Falloff rate of the rhythmicity magnitude (log10(sec))
    'f',  # Frequency of the rhymicity (Hz)
    'r',  # Rhythmicity
    's'  # Skipping [S.O.]
)))
ClimerPHat_len = 6

ClimerFit = namedtuple('ClimerFit', ', '.join((
    'theta_skipping',  # records whether theta_skipping was True or False in fit request
    'freq_trial',  # Firing frequency (Hz)
    'freq_trial_CI',  # 95% confidence interval for above
    'freq_win',  # Firing frequency in each window (Hz)
    'freq_win_CI',  # 95% confidence interval for above
    'hist',  # histogram counts of lags
    'bin_size_sec',  # bin_size_sec of histogram bins
    'LL_flat',  # Log liklihood of the arrhythmic fit
    'LL_noskip',  # Log liklihood of the non-skipping fit
    'LL_skip',  # Log liklihood of full fit with skipping [S.O.]
    'p_hat_flat',  # MLE estiamtors for on rhytmicity, see ClimerPHat
    'p_hat_noskip',  # MLE estimators for no theta-skipping see ClimerPHat
    'p_hat_skip',  # MLE estimators full see ClimerPHat [S.O.]
    'D_flat_v_noskip',  # Deviance of flat versus rhythmic fit ...
    'p_flat_v_noskip',  # ... and p value
    'D_noskip_v_skip',  # Deviamce of non-skippiong versus skipping fit [S.O.]...
    'p_noskip_v_skip',  # ... and p value [S.O]
)))


def statpoisci(m, lambdahat, alpha=0.05):
    """ ported from MATLAB's function of the same name.
    confidence interval for Poisson lambda parameter
    
    m and lambdahat should be broadcastable to the same shape.  """

    lsum = np.asanyarray(m * lambdahat)
    use_exact = lsum < 100
    lb, ub = np.empty(lsum.shape), np.empty(lsum.shape)

    # Chi-square exact method
    lb[use_exact] = chi2.ppf(alpha / 2, 2 * lsum[use_exact]) / 2
    ub[use_exact] = chi2.ppf(1 - alpha / 2, 2 * (lsum[use_exact] + 1)) / 2

    # Normal approximation
    lb[~use_exact] = norm.ppf(alpha / 2, lsum[~use_exact], sqrt(lsum[~use_exact]))
    ub[~use_exact] = norm.ppf(1 - alpha / 2, lsum[~use_exact], sqrt(lsum[~use_exact]))

    return (lb / m, ub / m)


def rhythmicity_pdf(p_hat, x, x_meaning='bin_inds',
                    theta_skipping=True, normalize=True,
                    as_log=False, compute_total_L=False,
                    bin_size_sec=0.001, max_lag_sec=0.6,
                    default_phat=[np.nan, np.nan, 1, 1, 0, 0],
                    ):
    """ Parametric distribution of lags for rhythmic neurons
    RHYTHMICITY_PDF generates the PMF of lags based on the parameters in p_hat.
    
    TAKES:
    p_hat - see ClimerPHat above for details.
            It can be a ClimerPHat/list/tuple/1d-array or a 2d-array.
            Iterating over it corresponds to iterating over a ClimerPHat instance.
            Thus, p_hat[0] corresponds to the first field in ClimerPHat etc.
            However it is permitted to be shorter than ClimerPHat, in such cases
            additional values are added using default_phat.
            The inidivual values may be scalars or vectors. If vectors they should
            all be of the same length, (although you may mix vectors and scalars).
            
    x, x_meaning - if x_meaning='bin_inds' then x is a full list of lags expressed 
             as bin indices.
             if x_meaning='hist' then x is a histogram of the bin_indices, 
             this is useful in combination with compute_total_L=True, as we
             can more efficiently sum over a histogram.
                 
    normalize - when true, normalizes the PDF to be a probability distribution 
                (The integral over the window is 1). If false, the value at 0 is 1.
    as_log -    when true, take logarithm of output. It is generally more efficient
                to do this inside the function.
                
    RETURNS:
       L: The liklihood or log-likilihood of each lag. 
       
    See top of module for notes on authorship etc.
    
    TODO: computing all the cosines is pretty slow. If you really want speed then
    it might be worth considering having a table of pre-computed values. You would
    round freq to your desired precision and then lookup or create the required
    cos(2 pi f t) and cos(pi f t) arrays.  It might be easier and possibly better
    to only deal with the case where f is a vector, and check to see if there are
    not that many distinct values. Could then construct a temproary cosine table.
    Could do the same for sqrt(1-s), athough that isn't as pressing.
    Note that this is slightly different to, though inspired by, the optimisation
    implemented in the original Matlab code.
    """

    # Force params to be a list of 1d vectors and/or scalars
    # Note vectors will need to be the same length, or an exception will occur later
    if isinstance(p_hat, np.ndarray):
        if p_hat.ndim == 1 or min(p_hat.shape) == 1:
            p_hat = list(p_hat)
        elif p_hat.ndim == 2:
            if p_hat.shape[0] > ClimerPHat_len:
                raise Exception("Got 2D p_hat but first dim is longer than number of params")
            p_hat = list(p_hat)
        else:
            raise Exception("p_hat must be 1D or 2D, not more.")
    else:
        p_hat = list(map(lambda p: np.asarray(p).ravel(), p_hat))

    # Pad with additional params if neccessary
    if len(p_hat) < ClimerPHat_len:
        p_hat += map(lambda x: np.asarray(x).ravel(), default_phat[len(p_hat):])

    # construct the namedtuple, adding an additional dimension to all arrays
    p_hat = ClimerPHat._make(map(lambda p: p[:, np.newaxis] if p.ndim > 0 else p, p_hat))

    if np.any(p_hat.b) < 0:
        raise Exception("p_hat's b value(s) must be non-negative.")

    if theta_skipping:
        F_t = lambda t: 1.0 / 4.0 * (
                    (2 + 2 * sqrt(1 - p_hat.s) - p_hat.s)
                    * cos(2 * pi * p_hat.f * t)
                    + 4 * p_hat.s * cos(pi * p_hat.f * t)
                    + 2
                    - 2 * sqrt(1 - p_hat.s)
                    - 3 * p_hat.s)
    else:
        F_t = lambda t: cos(2 * pi * p_hat.f * t)

    L_t = lambda t: (1 - p_hat.b) \
                    * exp(-t * 10 ** -p_hat.tau) \
                    * (p_hat.r
                       * exp(-t * 10 ** -p_hat.c.astype(float))
                       * F_t(t)
                       + 1) \
                    + p_hat.b

    # compute likelihood lookup table, which will be of 
    # shape [m x n_bins] where m is the number of paramater sets
    n_bins = np.ceil(max_lag_sec / bin_size_sec)
    bin_centres = np.arange(n_bins + 1) * bin_size_sec + bin_size_sec / 2
    L_bin = L_t(bin_centres[np.newaxis, :])
    if normalize:
        L_bin /= np.sum(L_bin, axis=1, keepdims=True)
    L_bin[(L_bin < 0) | np.isnan(L_bin) | (L_bin == np.inf)] = realmin

    if as_log:
        L_bin = np.log(L_bin)

    if x_meaning == 'bin_inds':
        # lookup likelihood for each of the data points
        return L_bin[:, x]
    elif x_meaning == 'hist':
        if compute_total_L is False:
            raise Exception("Providing a hist for x only makes sense in conjuction with" \
                            " computing the total likelihood, so set compute_total_L=True.")
        # Multiply each bin's likelhood by the number of counts in the bin
        # then sum over all bins
        return np.dot(L_bin, x.ravel())
    else:
        raise Exception("Unrecognised 'x_meaning'.")


# @profile
def theta_mle_climer(times, duration, max_lag_sec=0.6,
                     theta_skipping=True, bin_size_sec=0.001,
                     lb=ClimerPHat(-1, 0, -1, 1, 0, 0),
                     ub=ClimerPHat(1, 1, 1, 13, 1, 1),
                     full_output=True):
    """ Use maximum liklihood estimation to find rhythmicity parameters.

    See top of module for authorship notes etc.
    
    Takes:
        times -     spike times in seconds, sorted.
        duration -  trial duration in seconds
        max_lag_sec - as it says
        bin_size_sec - as it says
        theta_skipping - if false we fit without using thet theta-skipping part of the model
        full_output - if false we do not calculate histogram of freq trial/win
        
    Returns a ClimerFit namedtuple. see top of module for meaning of attributes.
    
    TODO: in minimize we want to be able to specify alpha =0.05
    """

    n_spikes = len(times)

    iend = times.searchsorted(times + max_lag_sec, side='right')
    counts_in_window = iend - np.arange(n_spikes) - 1
    b_ind = repeat_ind(counts_in_window)
    a_ind = count_to(counts_in_window) + 1 + b_ind
    x = times[a_ind] - times[b_ind]
    x_bin_inds = np.floor(x / bin_size_sec).astype(np.int32)

    if full_output:
        # Firing rate is the mean count per second.
        # Assume firing is Poisson, and get confidence interval on this mean value.
        freq_trial = float(n_spikes) / duration
        freq_trial_CI = statpoisci(duration, freq_trial)

        # Above was for whole trial, now do separtely for each window
        freq_win = counts_in_window / max_lag_sec
        freq_win_CI = statpoisci(max_lag_sec, freq_win)
        # Climer divides the above by freq_trial to get a "multiplier"    

    # make histrogram from bin inds
    hist = np.bincount(x_bin_inds, minlength=int(np.ceil(max_lag_sec / bin_size_sec) + 1))

    # Prepare the list of common inputs to rhythmicity_pdf and the main minimizer
    kwargs_for_pdf = dict(x=hist,
                          x_meaning='hist',
                          as_log=True,
                          compute_total_L=True,
                          max_lag_sec=max_lag_sec,
                          bin_size_sec=bin_size_sec)
    kwargs_for_minimize = dict(options={'disp': False},
                               method='L-BFGS-B')
    # Initial guess using particle swarm
    p_hat_0, _ = pso(
        lambda p_hat: -rhythmicity_pdf(p_hat, theta_skipping=False, **kwargs_for_pdf),
        lb=lb[:-1], ub=ub[:-1], swarmsize=75, maxiter=100, info=False, func_takes_multiple=True)

    # Solve for no-skipping            
    opt_res = minimize(lambda p_hat: -rhythmicity_pdf(p_hat, theta_skipping=False, **kwargs_for_pdf),
                       x0=p_hat_0, bounds=list(zip(lb[:-1], ub[:-1])), **kwargs_for_minimize)
    LL_noskip, p_hat_noskip = opt_res.fun, ClimerPHat(*np.append(opt_res.x, 0))

    # Arrhythmic fit, use p_hat_0  from above (i.e. non-skipping version)
    opt_res = minimize(lambda p_hat: -rhythmicity_pdf(p_hat, theta_skipping=False, **kwargs_for_pdf),
                       x0=p_hat_0[:2], bounds=list(zip(lb[:2], ub[:2])), **kwargs_for_minimize)
    LL_flat, p_hat_flat = opt_res.fun, ClimerPHat(*np.append(opt_res.x, [1, 1, 0, 0]))

    D_flat_v_noskip = 2 * (LL_noskip - LL_flat)
    p_flat_v_noskip = 1 - chi2.cdf(D_flat_v_noskip, len(lb) - 2)  # TODO: check this is right

    if theta_skipping:  # Need to do full fit...
        # Again, initial guess using particle swarm
        p_hat_0, _ = pso( \
            lambda p_hat: -rhythmicity_pdf(p_hat, theta_skipping=True, **kwargs_for_pdf),
            lb=lb, ub=ub, swarmsize=75, maxiter=100, info=False, func_takes_multiple=True)

        # Again, solve, but now for full, skipping case        
        opt_res = minimize(lambda p_hat: -rhythmicity_pdf(p_hat, theta_skipping=True, **kwargs_for_pdf),
                           x0=p_hat_0, bounds=list(zip(lb, ub)), **kwargs_for_minimize)
        LL_skip, p_hat_skip = opt_res.fun, ClimerPHat(*opt_res.x)

        D_noskip_v_skip = 2 * (LL_skip - LL_noskip)
        p_noskip_v_skip = 1 - chi2.cdf(D_noskip_v_skip, 1)

    # Collect everuything required for output
    locals_ = locals()
    return ClimerFit(**{k: locals_.get(k, None) for k in ClimerFit._fields})


def plot_rhythmicity_pdf(p_hat, max_lag_sec=0.6, bin_size_sec=0.001, ax=None):
    n_bins = np.ceil(max_lag_sec / bin_size_sec).astype(int)
    y = rhythmicity_pdf(p_hat,
                        np.arange(n_bins),
                        normalize=False,
                        theta_skipping=False if p_hat.s == 0 else True).ravel()

    if ax is None:
        plt.cla()
        plt.plot(np.arange(n_bins) * bin_size_sec + 0.5 * bin_size_sec, y, 'r-')


def plot_theta_mle_climer(times=None, duration=None, r=None, show_legend=True, **kwargs):
    """ Calls theta_mle_climer and produces
    a plot of the results.    
    You can provide times, duration, and kwargs for theta_mle_climer, 
    in which case we will run the fitting computation. Alternatively,
    you can provide r, the returned ClimerFit tuple, we can then plot
    this directly.
    
    This function returns the output of theta_mle_climer.    """

    if r is None:
        r = theta_mle_climer(times, duration, **kwargs)

    plt.cla()
    bin_lefts = np.arange(len(r.hist)) * r.bin_size_sec
    N = np.sum(r.hist)

    plt.bar(bin_lefts, r.hist, width=r.bin_size_sec, ec='none', color=[0.7] * 3)
    plt.xlabel('time since spike (s)')
    plt.ylabel('count')
    plt.xlim(0, bin_lefts[-1] + r.bin_size_sec)

    fit_flat = rhythmicity_pdf(r.p_hat_flat,
                               np.arange(len(r.hist)),
                               normalize=True,
                               theta_skipping=False).ravel()
    lg_flat, = plt.plot(bin_lefts + r.bin_size_sec / 2,
                        fit_flat * N, 'g-')

    fit_noskip = rhythmicity_pdf(r.p_hat_noskip,
                                 np.arange(len(r.hist)),
                                 normalize=True,
                                 theta_skipping=False).ravel()
    lg_noskip, = plt.plot(bin_lefts + r.bin_size_sec / 2,
                          fit_noskip * N, 'r-')

    lg = (lg_flat, lg_noskip)
    lg_keys = ('no rhythmicity', 'no skipping')
    maxy = 1.2 * fit_noskip[0] * N
    if r.theta_skipping:
        fit_skip = rhythmicity_pdf(r.p_hat_skip,
                                   np.arange(len(r.hist)),
                                   normalize=True,
                                   theta_skipping=True).ravel()
        lg_skip, = plt.plot(bin_lefts + r.bin_size_sec / 2,
                            fit_skip * N, 'b-')
        lg = lg + (lg_skip,)
        lg_keys = lg_keys + ('with skipping',)
        maxy = 1.2 * fit_skip[0] * N

    plt.ylim(0, maxy)
    if show_legend:
        plt.legend(lg, lg_keys)

    plt.text(0.97, 0.02, 'mean freq={:0.1f}Hz'.format(r.freq_trial),
             va='bottom', ha='right', weight='bold', transform=plt.gca().transAxes)
    plt.title('s={:0.2f} f={:0.1f} r={:0.2f}'.format(
        r.p_hat_skip.s, r.p_hat_skip.f, r.p_hat_skip.r))
    return r


"""
##############################################################################
"""


class TrialThetaMLEClimer(object):
    """this is a mixin for the trial class - see main readme"""

    @append_docstring(theta_mle_climer)
    def get_theta_hist_mle_climer(self, t=None, c=None, times=None,
                                  duration=None, **kwargs):
        """if you provide times instead of t and c, you must also give duration.
        both values should be in seconds."""
        if times is None:
            times = self.tet_times(t, c, as_type='s')
            duration = self.duration

        return theta_mle_climer(times, duration, **kwargs)

    @append_docstring(get_theta_hist_mle_climer)
    def plot_theta_hist_mle_climer(self, t=None, c=None, times=None,
                                   duration=None, **kwargs):
        """if you provide times instead of t and c, you must also give duration.
        both values should be in seconds."""
        if times is None:
            times = self.tet_times(t, c, as_type='s')
            duration = self.duration
        return plot_theta_mle_climer(times, duration, **kwargs)
