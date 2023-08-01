# -*- coding: utf-8 -*-
"""
Copy this file and rename it to just "tint_customise.py", you can
then setup the neccessary customisations. Changes to that file will be
ignored by git.
"""
import os
from definitions import DATA_ROOT_FREYA, DATA_ROOT_GUIFEN, load_experiment_info_guifen
from datetime import datetime

LOCAL_BASE = '/home/jgeerts/Data/'


def make_fn_getter_for_base(base):
    """
    RAT SPECIFIC NOTES
    r2192
    >>2014-10-01
    Note this animal has a typo in the file name for the track session indicates the wrong data - file 20140110 should be 20141001
    i.e. that is 1st Oct 2014 NOT 10th Jan 2014. Note that digit switch applies to all files for that session (i.e. track) but
    not to the other files from that day (i.e. sleepPost)

    >>2104-09-17
    Track1 and SleepPost sessions present
    -----

    r2217
    >>2014-12-13
    Screening, Track1 and SleepPost present

    >>2014-12-18
    Screening, Track1 and SleepPost present
    -----

    r2142
    >>2014-08-06
    Screening, Track1 and SleepPost present
    -----

    r2335
    >>2015-10-26
    Screening, Track1 and SleepPost present
    -----

    r2336
    >>2015-11-01
    >>2015-11-04

    -----
    r2337
    >>2015-11-27
    >>2015-12-01

    Args:
        base:

    Returns:

    """

    def inner(animal, year=None, month=None, day=None, trial=None, record_n=1):
        if animal in os.listdir(DATA_ROOT_GUIFEN):
            df = load_experiment_info_guifen()
            if trial is None:
                trial = 1
            # t = df[df.Animal == '{}'.format(animal)]['Trial{}'.format(trial)].iloc[record_n - 1]
            t = df[(df.Animal == animal) & (df.ExperimentN == day)]['Trial{}'.format(trial)].iloc[0]
            return os.path.join(DATA_ROOT_GUIFEN, '{}'.format(animal), t), "axona"
        elif animal.lower() in os.listdir(DATA_ROOT_FREYA):
            trial_types = ['screening', 'sleepPOST', 'track1']
            if trial is None:
                trial = trial_types[-1]
            session = '{}-{:02d}-{:02d}'.format(year, month, day)
            fn = '{}{:02d}{:02d}_{}_{}'.format(year, month, day, animal.capitalize(), trial)
            if datetime(year, month, day) == datetime(2014, 10, 1) and trial == 'track1':  # see docstring
                fn = '{}{:02d}{:02d}_{}_{}'.format(year, day, month, animal.capitalize(), trial)
            return os.path.join(DATA_ROOT_FREYA, animal.lower(), session, fn), "axona"
    return inner


def make_fn_getter_guifen(base):
    def inner(animal, year=None, month=None, day=None, trial=None, record_n=1):
        df = load_experiment_info_guifen()
        if trial is None:
            trial = 1
        t = df[(df.Animal == animal) & (df.ExperimentN == day)]['Trial{}'.format(trial)].iloc[0]
        return os.path.join(DATA_ROOT_GUIFEN, '{}'.format(animal), t), "axona"
    return inner


def get_moser_fn(animal, year=2004, month='required', day='required', trial=1):
    return (r"\\128.40.50.4\data\dmanson\0ther people\Moser\r"
            r"{animal:}\{animal:}-{day:02}{month:02}{year:02}{trial:02}").format(
        animal=animal, day=day, month=month, year=year % 1000,
        trial=trial), "moser_mat"


def get_npx_fn(base):
    pass

TRIAL_FILENAME_GETTERS = {
    'default': make_fn_getter_for_base(LOCAL_BASE),
    'guifen': make_fn_getter_guifen(LOCAL_BASE),
    'moser_mat': get_moser_fn,
    'npx': get_npx_fn
}

TRIAL_SAMPLE_LIST = [

    # idx=0
    dict(fn="C:\\Users\\daniel\\Desktop\\DATA\\2174\\2014-10-23\\2174 2014-10-23 1",
         cells="t15c5 t15c6 t14c1 t13c2 t11c2".split(" "),
         shape="rect", w=110, h=110, kind="axona")

    ,  # idx=1
    dict(fn="C:\\Users\\daniel\\Desktop\\DATA\\2142\\2014-08-30\\2142 2014-08-30 4",
         cells="t3c1 t4c1".split(" "),
         shape="circ", w=125, h=125, kind="axona")

]
