# -*- coding: utf-8 -*-
from warnings import warn
import datetime
import inspect
import sys

# core imports
from core.pos_post import PosProcessor
from core import trial_caching
from core import trial_axona
from core import trial_moser_mat
from core import trial_basic_analysis
from core import trial_npx

# non-core imports
from noncore import trial_basic_plotting
from noncore import trial_summary
from noncore import trial_behaviour
from noncore import trial_gc
from noncore import trial_gc_plotting
from noncore import trial_warp
from noncore import trial_theta_mle_climer
from noncore import trial_freq
from noncore import trial_mle_drift
from noncore import trial_fit_ideal_grids
from noncore import trial_transform
from noncore import trial_gc_1d_sac

# stuff for customizing behaviour of the functions within this module
import tint_customise

__all__ = ['TrialAxonaAll', 'TrialMoserMatAll', 'get_trial', 'get_sample_trial',
           'PosProcessor']


# Mix together all the core and analysis to make an axona class..
class TrialAxonaAll(
    trial_caching.TrialCaching,
    trial_axona.TrialAxona,
    trial_basic_analysis.TrialBasicAnalysis,
    trial_basic_plotting.TrialBasicPlotting,
    trial_behaviour.TrialBehaviour,
    trial_summary.TrialSummary,
    trial_gc.TrialGCAnalysis,
    trial_gc_plotting.TrialGCAnalysisPlotting,
    trial_warp.TrialWarp,
    trial_theta_mle_climer.TrialThetaMLEClimer,
    trial_freq.TrialFreqAnalysis,
    trial_mle_drift.TrialMLEDrift,
    trial_fit_ideal_grids.TrialIdealGrid,
    trial_transform.TrialTransform,
    trial_gc_1d_sac.TrialGC1dSACProps):
    """ mixes together all the classes you want """

    def __init__(self, *args, **kwargs):
        trial_caching.TrialCaching.__init__(self)
        trial_axona.TrialAxona.__init__(self, *args, **kwargs)


# And to make a moser-mat class... (TODO: might want a base analysis class rather than repeating the above)
class TrialMoserMatAll(
    trial_caching.TrialCaching,
    trial_moser_mat.TrialMoserMat,
    trial_basic_analysis.TrialBasicAnalysis,
    trial_basic_plotting.TrialBasicPlotting,
    trial_behaviour.TrialBehaviour,
    trial_summary.TrialSummary,
    trial_gc.TrialGCAnalysis,
    trial_gc_plotting.TrialGCAnalysisPlotting,
    trial_warp.TrialWarp,
    trial_theta_mle_climer.TrialThetaMLEClimer,
    trial_freq.TrialFreqAnalysis,
    trial_mle_drift.TrialMLEDrift,
    trial_fit_ideal_grids.TrialIdealGrid,
    trial_transform.TrialTransform,
    trial_gc_1d_sac.TrialGC1dSACProps):
    """ mixes together all the classes you want """

    def __init__(self, *args, **kwargs):
        trial_caching.TrialCaching.__init__(self)
        trial_moser_mat.TrialMoserMat.__init__(self, *args, **kwargs)


class TrialNPXAll(
    trial_caching.TrialCaching,
    trial_npx.TrialNPX,
    trial_basic_analysis.TrialBasicAnalysis,
    trial_basic_plotting.TrialBasicPlotting,
    trial_behaviour.TrialBehaviour,
    trial_summary.TrialSummary,
    trial_gc.TrialGCAnalysis,
    trial_gc_plotting.TrialGCAnalysisPlotting,
    trial_warp.TrialWarp,
    trial_theta_mle_climer.TrialThetaMLEClimer,
    trial_freq.TrialFreqAnalysis,
    trial_mle_drift.TrialMLEDrift,
    trial_fit_ideal_grids.TrialIdealGrid,
    trial_transform.TrialTransform,
    trial_gc_1d_sac.TrialGC1dSACProps):

    def __init__(self, *args, **kwargs):
        trial_caching.TrialCaching.__init__(self)
        trial_npx.TrialNPX.__init__(self, *args, **kwargs)


def get_trial(a, y=None, m=None, d=0, t=None, p='default'):
    """This uses TRIAL_FILENAME_GETTERS in trials_database.py to lookup trial
    file names in an easy to use way.
    
    At minimum you must specify animal, a. 
    
    d can be a tuple of (y, m, d), or you can specify them individually. If
    you do specify them individually the current year and month will be used
    by default so you can ommit them.  d can also be zero or negative meanign
    that many days in the past.
    
    t is trial identifier, can be None.
    
    p is used to select between multiple getter functions, which probably 
    correspond to different people's data, or various file-naming conventions
    you might have used in the past with your own data. You can also use it
    to distinguish between local datasets and datasets stored on a server.
    
    Examples::
        
        get_trial(a=2219) # gets the default trial for today from rat 2219
        get_trial(a=2219, d=-1, t=3) # gets trial 3 from yesterday from rat 2219
        get_trial(a=2219, d=24) # gets the default trial from 24th of this month from rat 2219
        get_trial(a=2219, d=17, m=5, t=2) # gets trial 2 from 17th May of this year for rat 2219
        get_trial(a=2219, d=(2015, 3, 21), t=6) # gets trial 6 from 21st March 2015 for rat 2219
        get_trial(a=2051, p='charlie') # gets default trial for charlie's rat 2051 from today
        
    """
    if p not in tint_customise.TRIAL_FILENAME_GETTERS:
        raise KeyError("No filename getter associated with key '" + p + "'.")

    getter = tint_customise.TRIAL_FILENAME_GETTERS[p]

    if isinstance(d, (list, tuple)):
        if len(d) != 3:
            raise ValueError("d is either of the form (y,m,d), or just a day "
                             "of the month, or a zero/negative day offset from"
                             " today.")
        y, m, d = d
    elif d < 1:
        if y is not None or m is not None:
            raise ValueError("If you use d<=0, you cannot specify y or m.")
        then = datetime.datetime.now() + datetime.timedelta(days=d)
        y, m, d = then.year, then.month, then.day
    elif None in (y, m):
        now = datetime.datetime.now()
        if y is None:
            y = now.year
        if m is None:
            m = now.month

    if y < 100:
        y += 2000
    if m < 1 or m > 12:
        raise ValueError("Month should be on the interval [1, 12].")
    if d < 1 or d > 31:
        raise ValueError("Day should be on the interval [1, 31].")

    fn, kind = getter(animal=a, year=y, month=m, day=d, trial=t)
    if kind == "axona":
        return TrialAxonaAll(fn=fn, cut_file_pattern='.cut')
    elif kind == "moser_mat":
        return TrialMoserMatAll(fn=fn)
    else:
        raise ValueError("unrecognised trial kind: '" + kind + "'.")


def get_sample_trial(idx=0):
    """ Indexes into the `TRIAL_SAMPLE_LIST` in trials_database.py, and returns
    a constructed `Trial` object.
    """
    ls = tint_customise.TRIAL_SAMPLE_LIST
    if idx >= len(ls):
        warn("only {} sample trials in the list, returning 0th sample.".format(len(ls)))
        idx = 0

    samp = ls[idx]
    if 'cells' in samp:
        print("nice cells: ", ', '.join(samp['cells']))

    if 'kind' not in samp or samp['kind'] == 'axona':
        pos_proc_kwargs = inspect.getargspec(PosProcessor.__init__).args
        alias_dict = dict(shape='fit_shape', w='fixed_w', h='fixed_h')
        tr = TrialAxonaAll(fn=samp['fn'],
                           pos_processor=PosProcessor(**{
                               alias_dict.get(k, k): v for k, v in samp.items()
                               if alias_dict.get(k, k) in pos_proc_kwargs}))
    elif samp['kind'] == 'moser_mat':
        pass
    else:
        raise ValueError("unrecognised Trial kind {}".format(samp['kind']))

    return tr
