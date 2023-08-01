# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import re
import subprocess
import sys
sys.path.append('.')

from warnings import warn
from scipy.ndimage.filters import maximum_filter1d
from utils.custom_casting import str2float, str2int

from core.pos_post import PosProcessor
from core.read_axona_file import read_axona_file


DEFAULT_POS_PROCESSOR_AXONA = PosProcessor()
POS_NAN = 1023

        
class TrialAxona(object):
    """
    Implements the Trial class as described in readme.md.
    
    **Additional almost-public methods/attributes etc.**:   
    To select between cuts use TrialAxona._cut_file_names. This is 
    a CutFileChooser instance which offers a variety of handy tools.
    You can pass a cut_file_pattern through to this in the TrialAxona 
    constructor.  If you change cuts, the next time you request anything
    for a specific cell on a tetrode you will get the new cut, i.e. there
    should not be anything cached to do with cuts (except the loaded
    cut file itself).
    You can change `_pos_processor` simply by setting it. This will clear all
    previously loaded pos data (except the header).  Note that nothing outside
    this file should cache stuff that uses pos (unless it is willing to deal
    with this issue).
    
    **Convention**:   
    attributes ending in _cache are/contain things which can be deleted
    and recreated by reading off disk using parameters/filenames also
    stored as attributes.
    
    TODO: deal with pos_shape stuff and eeg choices        
    """
    
    """ implementation-specific stuff """
    
    def __init__(self, fn, pos_processor=None, cut_file_pattern=None):
        if fn.lower().endswith(".set"):
            fn = fn[:-4]
 
        self._pos_processor_active = pos_processor or DEFAULT_POS_PROCESSOR_AXONA
        self.recording_type = 'tet'
        self._fn = fn
        self._path, self._experiment_name = os.path.split(fn)
        _, self._animal_name = os.path.split(self._path)
                
        # These specify the currently chosen cut/eeg files.
        self._cut_file_names = CutFileChooser(fn, cut_file_pattern) # dict-like, with tet nums as keys
        self._eeg_file_name = fn + ".eeg" # TODO: need something a bit like CutFileChooser..also think about theta* in basicAnalysis
        self._egf_file_name = fn + ".egf"
        
    def __str__(self):
        return "<TrialAxona instance: fn='" + self._fn + "'>"

    def __repr__(self):
        return str(self)
        
    def open_folder(self):
        """launches windows explorer at the relevant folder.
        TODO: support other OSes."""
        cmd = 'explorer "{}"'.format(os.path.split(self._fn)[0])
        print(cmd)
        subprocess.Popen(cmd)

    @property
    def _pos_processor(self):
        return self._pos_processor_active
    
    @_pos_processor.setter
    def _pos_processor(self, pp):
        self._pos_processor_active = pp
        self._clear_cache(drop=('pos',))
        
    @property
    def _set_header(self):
        self._load_into_cache_set()
        return self._cache_set_header

    @property
    def _pos_header(self):
        self._load_into_cache_pos(header_only=True)
        return self._cache_pos_header

    def _tet_header(self, t):
        self._load_into_cache_tet(t, header_only=True)
        return self._cache_tet_header[t]
    
    @property
    def _eeg_header(self):
        self._load_into_cache_eeg(self._eeg_file_name, header_only=True)
        return self._cache_eeg_header[self._eeg_file_name]

    @property
    def _egf_header(self):
        self._load_into_cache_egf(self._egf_file_name, header_only=True)
        return self._cache_egf_header[self._egf_file_name]

    @property
    def has_egf(self):
        return os.path.exists(self._egf_file_name)

    @property        
    def _n_leds(self):
         hd = self._set_header
         return int(sum( [str2float(hd['colactive_' + x]) for x in ['1','2']]))
                        
    def _cut(self, tet_num):
        fn = self._cut_file_names[tet_num]
        self._load_into_cache_cut(fn)
        return self._cache_tet_cut[fn]
        
    def _available_cells(self, tet_num):
        """returns unique(cut) for the given tetrode."""
        return np.unique(self._cut(tet_num))

    def get_available_cells(self, tet_num, include_mua=False):
        try:
            available_cells = self._available_cells(tet_num)
        except KeyError:
            return np.array([])
        if include_mua:
            return available_cells
        else:
            return available_cells[np.nonzero(available_cells)]

    def _load_into_cache_set(self):
        if not self._cache_has('set', '_cache_set_header'):
            self._cache_set_header, _ = read_axona_file(self._fn + ".set", False)       
        
    def _load_into_cache_cut(self, fname):
        if self._cache_has('tet', '_cache_tet_cut', fname):
            return
            
        if not os.path.isfile(fname):
            warn("Cut file not found: %s" % (fname))
            self._cache_tet_cut[fname] = np.array([])
            return
        
        with open(fname,'r') as f:
            data = f.read()
            f.close()
        if fname.endswith(".cut"):
            data = data.split('spikes: ', 1)[1]
        # [.cut] data is now: n_spikes\n# # # # # etc.
        # [.clu] data is now: n_clusters\n#\n#\n#\n#\n etc.
        
        self._cache_tet_cut[fname] = np.fromiter(map(int, data.split()[1:]), dtype=int)  # changed python 3 compatible
        self._cache_tet_cut[fname].setflags(write=False)
            
    def _load_into_cache_eeg(self, f, header_only=False): 
        bad_tol = 2 # how close to max/min before considering it saturated
        bad_window_s = 1.5 # how many seconds to extend "bad" either side of saturation
        
        if header_only:
            if self._cache_has('eeg', '_cache_eeg_header', f):
                return
        else:
            if self._cache_has('eeg', ['_cache_eeg_header', '_cache_eeg',
                                       '_cache_eeg_is_bad'], f):
                return
        
        self._cache_eeg_header[f], _ = read_axona_file(f, None, True)
        if not self._cache_eeg_header[f]:
            raise IOError("No header found, the file itself probably doesn't exist.\n{}".format(f))
        if header_only:
            return
            
        hd = self._cache_eeg_header[f]
        byte_per_samp = str2float(hd.get('bytes_per_sample','1'))
        fmt = '=b' if byte_per_samp == 1 else '=h' # TODO: check high samp rate 
        _, data = read_axona_file(f, [('eeg',fmt)])
        
        n_samps = str2int(hd.get('num_EEG_samples', hd.get('num_EGF_samples')))
        eeg = data['eeg'][:n_samps] # this is important as there can be nonsense at the end of the file.
        eeg_range = np.iinfo(eeg.dtype)
        is_bad = (eeg<=eeg_range.min+bad_tol) | (eeg>=eeg_range.max - bad_tol)  
        is_bad = maximum_filter1d(is_bad, 
                                  int(self.eeg_samp_rate*bad_window_s)*2+1)
        eeg = eeg.astype(np.single)
        eeg.setflags(write=False)
        self._cache_eeg[f] = eeg

        if np.any(is_bad):
            self._cache_eeg_is_bad[f] = is_bad
            self._cache_eeg_is_bad[f].setflags(write=False)
        else:
            self._cache_eeg_is_bad[f] = False
        warn('NotImplemented: EEG conversion to volts.')

    def _load_into_cache_egf(self, f, header_only=False):
        bad_tol = 2  # how close to max/min before considering it saturated
        bad_window_s = 1.5  # how many seconds to extend "bad" either side of saturation

        if header_only:
            if self._cache_has('egf', '_cache_egf_header', f):
                return
        else:
            if self._cache_has('egf', ['_cache_egf_header', '_cache_egf',
                                       '_cache_egf_is_bad'], f):
                return

        self._cache_egf_header[f], _ = read_axona_file(f, None, True)
        if not self._cache_egf_header[f]:
            raise IOError("No header found, the file itself probably doesn't exist.\n{}".format(f))
        if header_only:
            return

        hd = self._cache_egf_header[f]
        byte_per_samp = str2float(hd.get('bytes_per_sample', '1'))
        fmt = '=b' if byte_per_samp == 1 else '=h'  # TODO: check high samp rate
        _, data = read_axona_file(f, [('eeg', fmt)])

        n_samps = str2int(hd.get('num_EEG_samples', hd.get('num_EGF_samples')))
        eeg = data['eeg'][:n_samps]  # this is important as there can be nonsense at the end of the file.
        eeg_range = np.iinfo(eeg.dtype)
        is_bad = (eeg <= eeg_range.min + bad_tol) | (eeg >= eeg_range.max - bad_tol)
        is_bad = maximum_filter1d(is_bad,
                                  int(self.egf_samp_rate * bad_window_s) * 2 + 1)
        eeg = eeg.astype(np.single)
        eeg.setflags(write=False)
        self._cache_egf[f] = eeg

        if np.any(is_bad):
            self._cache_egf_is_bad[f] = is_bad
            self._cache_egf_is_bad[f].setflags(write=False)
        else:
            self._cache_egf_is_bad[f] = False
        warn('NotImplemented: EEG conversion to volts.')

    def _load_into_cache_pos(self, header_only=False):
        if header_only:
            if self._cache_has('pos','_cache_pos_header'):
                return
        else:
            if self._cache_has('pos', ['_cache_pos_header', '_cache_xy',
                '_cache_w', '_cache_h', '_cache_pos_shape', '_cache_dir']):
                return
        
        self._cache_pos_header, data = read_axona_file(self._fn + ".pos", 
                                                       [('ts','>i'),('pos','>8h')], header_only)
        if not self._cache_pos_header:
            raise IOError('Could not find pos file header/data.')
        if header_only:
            return
                         
        if self._n_leds == 1:
            led_pos = np.ma.masked_values(data['pos'][:,0:2], value=POS_NAN)
            led_pix = np.ma.masked_values(data['pos'][:,4:5], value=POS_NAN)
        elif self._n_leds == 2:
            led_pos = np.ma.masked_values(data['pos'][:,0:4], value=POS_NAN)
            led_pos.shape = -1, 2, 2
            led_pix = np.ma.masked_values(data['pos'][:,4:6], value=POS_NAN)
            
        self._cache_xy, self._cache_dir , _, self._cache_w, self._cache_h, self._cache_pos_shape = \
            self._pos_processor(led_pos, led_pix, self.pos_samp_rate, self._pos_header, self._n_leds)        
        if isinstance(self._cache_dir, np.ndarray):
            self._cache_dir.setflags(write=False)
        self._cache_xy.setflags(write=False)
        
    def _load_into_cache_tet(self, num, waves=False, header_only=False):
        if header_only:
            if self._cache_has('tet', '_cache_tet_header', num):
                return
        elif self._cache_has('tet', ['_cache_tet_times', '_cache_tet_header'], num):
            return

        if header_only:
            self._cache_tet_header[num],_ = read_axona_file(self._fn + "." + str(num), None, True)
            return
            
        self._cache_tet_header[num], data = read_axona_file(self._fn + "." + str(num),
                                                            [('ts','>i'),('waveform','50b')])
        self._cache_tet_times[num] = data['ts'][::4]
        self._cache_tet_times[num].setflags(write=False)
        
        if waves:
            raise NotImplementedError # TODO: note we were hoping to be able to avoid loading waves entierly...but I don't think that's possible

        
    """ ######################################################## 
        ######################################################## """

    @property
    def path(self):
        return self._path

    @property
    def animal_name(self):
        return self._animal_name

    @property
    def experiment_name(self):
        return self._experiment_name
        
    @property
    def duration(self):
        return str2float(self._set_header['duration'])
    
    """ Position related stuff """
    
    @property
    def n_pos(self):
        """returns number of pos samples"""
        return str2float(self._pos_header['num_pos_samples'])
    
    @property
    def pos_samp_rate(self):
        """returns position sampling rate in Hz"""
        return str2float(self._pos_header['sample_rate'])
        
    @property
    def xy(self):
        """returns [2xn_pos] values in centimeters, x values are [0,w], 
            y values are [0, h]."""
        self._load_into_cache_pos()
        return self._cache_xy

    @property
    def w(self):
        """ returns arena width in cm"""
        self._load_into_cache_pos()
        return self._cache_w

    @property
    def h(self):
        """ returns arena width in cm"""
        self._load_into_cache_pos()
        return self._cache_h
        
    @property
    def pos_shape(self):
        self._load_into_cache_pos()
        return self._cache_pos_shape
            
    @property
    def speed(self):
        """returns values in centimeters per second, lem=n_pos"""
        if not self._cache_has('pos','_cache_speed'):
            diff_xy = np.diff(self.xy, axis=1)
            if len(diff_xy):
                self._cache_speed = np.hypot(diff_xy[0], diff_xy[1])
                self._cache_speed = np.append(self._cache_speed,[0])
                self._cache_speed = self._cache_speed * self.pos_samp_rate
            else:
                self._cache_speed = np.array([])
            self._cache_speed.setflags(write=False)
        return self._cache_speed

    @property
    def dir(self):
        """ returns direction of facing in radians len=n_pos,
        i.e. using all availalbe LEDs, only falling back on displacement vector
        if 1LED was used."""
        self._load_into_cache_pos()
        if isinstance(self._cache_dir, np.ndarray):
           return self._cache_dir
        else:
           return self.dir_disp
           
    @property
    def dir_disp(self):
        """returns direction of displacement vector in radians, len=n_pos"""
        if not self._cache_has('pos', '_cache_dir_disp'):
            xy = self.xy
            self._cache_dir_disp = np.arctan2(np.ediff1d(xy[0,:], to_end=[0]),
                                              np.ediff1d(xy[1,:], to_end=[0]))
            self._cache_dir_disp.setflags(write=False)
        return self._cache_dir_disp      
    
    """ Spike-related stuff """

    def spk_times(self, t, c=None, as_type='s', acceptable_bleed_time=1.0,
                  min_spike_count=1):
        """ 
        Gets the times for tetrode number ``t``.  If ``c`` is not none, it
        returns only the times for the cell on the given tetrode.
        
        ``asType`` can take one of the following values:
        
        * ``'x'`` - do not convert, i.e. units are given by ``self.tetTimebase[t]``
        * ``'s'`` - seconds (returned as double,all others are ints)
        * ``'p'`` - pos index
        * ``'e'`` - eeg index
        
        ``acceptable_bleed_time`` - a value in seconds.  If the last few spikes times 
        are beyond the end of the trial by less than this amount they will silently
        be removed.  If they go beyond the end of the trial by more than this amount
        an error will be thrown.   (Set to 0/Infinity to throw error for any bleed or
        swallow all extra spikes.)
    
        ``min_spike_count`` - if the number of spikes is less than this an
        error will be thrown.
        """
        if t is None:
            raise ValueError("you must at least specify tetrode number")
        self._load_into_cache_tet(t)
        times = self._cache_tet_times[t]       
        timebase = str2int(self._tet_header(t)['timebase'])

        if c is not None:
            cut = self._cut(t)
            times = times[cut==c] 
            if len(times) == 0:
                raise ValueError("No spikes found for cell in cut.")
            if len(times) < min_spike_count:
                raise ValueError("More than zero, but fewer than min_spike_count spikes found for cell in cut.")
            
        if not len(times):
            return times
            
        # check for spikes that bleed beyond the end of the trial
        if times[-1] > self.duration*timebase:
            # If they only bleed a little beyond the end of the trial we can just discard them, otherwise we should raise an error
            if times[-1] > (self.duration + acceptable_bleed_time)*timebase:
                raise Exception(("At least one spike time is more "
                                 "than {:0.2f}s beyond the end of the trial").format(acceptable_bleed_time))
            else:
                times = times[:np.searchsorted(times,self.duration*timebase)]
                    
        if as_type == 'x':
            pass
        elif as_type == 's':
            times = times.astype(np.double) / timebase
        elif as_type == 'p':
            factor = self.pos_samp_rate/timebase
            times = (times*factor).astype(int) #note this is rounding down
        elif as_type == 'e':
            factor = self.eeg_samp_rate/timebase
            times = (times*factor).astype(int) #note this is rounding down
        else:
            raise Exception("unknown type flag for tetTimes: %s" % (as_type))
   
        return times

    """ EEG-related stuff """

    @property
    def eeg_samp_rate(self):
        """ value in Hz. """
        return str2float(self._eeg_header['sample_rate'])

    @property
    def egf_samp_rate(self):
        """ value in Hz. """
        return str2float(self._egf_header['sample_rate'])

    def eeg(self, bad_as='nan', filetype='eeg'):
        if filetype == 'eeg':
            fn = self._eeg_file_name
        elif filetype == 'egf':
            fn = self._egf_file_name
        else:
            raise ValueError('Not a known EEG file format.')
        """ gives the eeg values in volts. """
        self._load_into_cache_eeg(fn)
        if bad_as is None or self.eeg_is_bad is None:
            return self._cache_eeg[fn]
        else:
            return np.where(self.eeg_is_bad, np.nan if bad_as == 'nan' else bad_as,
                            self._cache_eeg[fn])
            # TODO: we should dampen eeg either side, which makes for better filtering

    def egf(self, bad_as='nan'):
        """ gives the eeg values in volts. """
        fn = self._egf_file_name
        self._load_into_cache_egf(fn)
        if bad_as is None or self.egf_is_bad is None:
            return self._cache_egf[fn]
        else:
            return np.where(self.egf_is_bad, np.nan if bad_as == 'nan' else bad_as,
                            self._cache_egf[fn])
            # TODO: we should dampen eeg either side, which makes for better filtering

    @property
    def eeg_is_bad(self):
        """ either None (if eeg is perfect) or ``len(eeg), dtype=bool``
        True where eeg is bad, probably because of saturation."""
        self._load_into_cache_eeg(self._eeg_file_name)
        ret = self._cache_eeg_is_bad[self._eeg_file_name]
        return ret if isinstance(ret, np.ndarray) else None # couldn't use None for caching reasons

    @property
    def egf_is_bad(self):
        """ either None (if eeg is perfect) or ``len(eeg), dtype=bool``
        True where eeg is bad, probably because of saturation."""
        self._load_into_cache_egf(self._egf_file_name)
        ret = self._cache_egf_is_bad[self._egf_file_name]
        return ret if isinstance(ret, np.ndarray) else None # couldn't use None for caching reasons

    def get_available_tets(self):
        self._cut_file_names._populate_cache()  # TODO: get cut file names
        return sorted(list((tet for tet in self._cut_file_names._cache.keys())))


CUT_FILENAME_REGEX = re.compile('_(\d+)\.cut$|\.clu\.(\d+)$')

class CutFileChooser(object):
    """ A class to help with chosing between cut files.
    `set_fn` is the full path to a set file.
    `pattern` can be `None`, in which case the most
    recently modified auto-discovered cut file will be used.
    If you can also specify to limit choice to clu/cut using the string
    `.cut` or `.clu`.
    Note that checks for recent modifications are done once, to refresh
    you would need to manually call refresh.
    Alternatively, specify a full pattern, which is a string including one or
    more of the following tokens:
    ``%PATH%``, ``%EXPERIEMENT%`` and ``%TET%``. The tokens will be replaced
    with their respective values in order to obtain the correct cutfile name.    
    """
    def __init__(self, fn, pattern):
        self._fn = fn       
        self._path, self._experiment_name = os.path.split(fn)
        self._cache = {} # key=tet_num
        self._has_used_setter = False
        self._pattern = pattern
        
    def _get_all_trial_files(self):
         # get list of files, sorted from oldest to newest
        fnames = glob.glob(self._fn + "*") 
        sorted(fnames, key=os.path.getmtime)
        return fnames

    @staticmethod
    def _parse_filename(f):
        m = CUT_FILENAME_REGEX.search(os.path.split(f)[1])
        if m is not None:
            if m.group(1) is not None:
                return '.cut', int(m.group(1))
            else:
                return '.clu', int(m.group(2))
        return None, None

    def _populate_cache(self):
        if self._pattern not in  ('.cut', '.clu', None):
            raise Exception("don't call populate cache when there is a pattern.")
        if self._has_used_setter:
            raise NotImplementedError("cannot populate cache after manually "
                                      "choosing something.")
            
        fnames = self._get_all_trial_files()        
        for f in fnames:
            fmt, tet_num = self._parse_filename(f)
            if tet_num is None:
                continue
            if self._pattern is None or self._pattern == fmt:
                self._cache[tet_num] = f

    def pick(self, t):
        """ provides a list of available cuts and a command line to select between them"""
        fnames = self._get_all_trial_files()
        cut_files = []
        for f in fnames:
            fmt, tet_num = self._parse_filename(f)
            if tet_num is None:
                continue
            cut_files.append(f)
            print("[{}] {}".format(len(cut_files), f))
        if not len(cut_files):
            print("No cut files found for " + self._fn)
            return
        print("Note that other cut/clu files may exist but were not automatically found.")
        print("You can explicitly set such files using CutFileChooser[{}] = 'path'".format(t))
        print("Select cut file for tetrode #{} from above list, or -1 to cancel.".format(t))

        idx = input("Choice: ")
        idx = str2int(idx)
        if idx == -1:
            print("canceled")
            return
        if idx < 1 or idx > len(cut_files):
            raise ValueError("bad choice")
        self.__setitem__(t, cut_files[idx-1])
        print("selected: " + cut_files[idx-1])
        
    def __getitem__(self, tet_num):
        if tet_num not in self._cache:
            if self._pattern in ('.cut', '.clu', None):
                self._populate_cache()
            else:
                self._cache[tet_num] = self._pattern.replace("%PATH%", self._path) \
                                                    .replace("%EXPERIMENT%", self._experiment_name) \
                                                    .replace("%TET%", str(tet_num))
        return self._cache[tet_num]
    
    def __setitem__(self, tet_num, fn):
        if self._pattern:
            warn("overriding cut pattern is confusing.")
        self._cache[tet_num] = fn
        self._has_used_setter = True
    
    def set_pattern(self, pattern):
        self._pattern = pattern
        self.refresh()
        
    def refresh(self):
        if self._has_used_setter:
            warn("clearing previous explicitly chosen cut file names")
        self._cache = {}
        self._has_used_setter = False
        
    def __str__(self):

        if self._pattern not in (None, '.clu', '.cut'):
            if self._has_used_setter:
                return "CutFileChooser pattern: '" + self._pattern + "'\n" +\
                        "note that some items may have been selected manually>"
            else:
                return "CutFileChooser pattern: '" + self._pattern + "'"
        else:
            if not len(self._cache):
                self._populate_cache()
            return "CutFileChooser chosen {}: \n\t".format(
                    "most recent" if self._pattern is None else
                    "most recent " + self._pattern
                ) + "\n\t".join(
                "{:2}: {}".format(k,v) for k, v in self._cache.items()            
                ) + ("" if not self._has_used_setter else 
                     "\nnote that some items may have been selected manually.")
            
    def __repr__(self):
        return str(self)