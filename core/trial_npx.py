import numpy as np
import os
import os.path as op
import mat73
import pandas as pd


class TrialNPX(object):
    """Implements the Trial class as described in readme.md for neuropixel data in matlab format.
    """
    def __init__(self, fn):

        self.recording_type = 'npx'
        self._fn = fn
        self._path, self._experiment_name = os.path.split(fn)
        path, self._sess = os.path.split(self._path)
        _, self._animal_name = os.path.split(path)

        data = mat73.loadmat(fn)
        self._spk = data['npx']['spk']
        self._pos = data['npx']['pos']
        self._lfp = data['npx']['lfp']
        self._cut = data['npx']['spk']['clu']

        self.pos_samp_rate = self._pos['settings']['posSampleRate']
        self.eeg_samp_rate = np.round(1 / (self._lfp['times'][1] - self._lfp['times'][0]))

        # initialise VR trial info stuff (TODO: maybe move this stuff to a separate mixin)
        self._fixed_xy = None
        self._f_offset = None
        self._trial_idx = None
        self._trial_info = None

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def xy(self):
        return self._pos['xy'].T

    @property
    def duration(self):
        return self._pos['times'][-1] - self._pos['times'][0]

    @property
    def start_time(self):
        return self._pos['times'][0]

    def spk_times(self, t=None, c=None, as_type='s', min_spike_count=1):
        """Gets the spike times for a given cell c.

        Args:
            t: Has to be None. Only here to play nicely with Axona trial classes.
            c:
            as_type:
            min_spike_count:

        Returns:

        """
        if t is not None:
            raise ValueError('Tetrodes do not apply here.')

        if as_type == 's':
            times = self._spk['times']
        elif as_type == 'p':
            times = self._spk['posSample'].astype(int)
        else:
            raise Exception("unknown type flag for tetTimes: %s" % as_type)

        if c is not None:
            cut = self._spk['clu']
            times = times[cut == c]
            if len(times) == 0:
                raise ValueError("No spikes found for cell in cut.")
            if len(times) < min_spike_count:
                raise ValueError("More than zero, but fewer than min_spike_count spikes found for cell in cut.")

        return times

    def get_available_cells(self, t=None):
        return self._spk['cids'].astype(int)

    @property
    def animal_name(self):
        return self._animal_name

    @property
    def session(self):
        return self._sess

    @property
    def n_pos(self):
        return self._pos['times'].size

    @property
    def w(self):
        return self._pos['settings']['RightBorder']

    @property
    def h(self):
        return self._pos['settings']['TopBorder']

    @property
    def speed(self):
        return self._pos['speed']

    @property
    def dir(self):
        return self._pos['dir']

    @property
    def pos_settings(self):
        return self._pos['settings']

    @property
    def pos_times(self):
        """Return times in seconds corresponding to position samples.
        """
        return self._pos['times']

    def eeg(self):  # FIXME: get the LFP traces for these data.
        raise NotImplementedError('There seems to be no LFP data!')

    @property
    def dir_disp(self):
        """returns direction of displacement vector in radians, len=n_pos"""
        if not self._cache_has('pos', '_cache_dir_disp'):
            xy = self.xy
            self._cache_dir_disp = np.arctan2(np.ediff1d(xy[0, :], to_end=[0]),
                                              np.ediff1d(xy[1, :], to_end=[0]))
            self._cache_dir_disp.setflags(write=False)
        return self._cache_dir_disp

    @property
    def pos_shape(self):
        return None

    @property
    def path(self):
        return self._path

    def task_epoch(self):
        reward_xy = self._pos['vr']['RewardSpawned']['Data'].T
        reward_times = self._pos['vr']['RewardSpawned']['TimeStamps']

        # identify the fixed reward location and corresponding trials
        reward_x, n_returns = np.unique(reward_xy[0], return_counts=True)
        fixed_x = reward_x[n_returns == n_returns.max()]
        fixed_trials = np.where(reward_xy[0] == fixed_x)[0]
        assert np.isclose(reward_xy[:, fixed_trials].T, reward_xy[:, fixed_trials[0]]).all()
        self._fixed_xy = reward_xy[:, fixed_trials[0]]

        # find the trial offset type for each trial
        self._trial_idx = np.arange(len(reward_times))
        self._f_offset = np.empty(len(reward_times))
        self._f_offset[:] = np.nan
        for t in range(len(self._trial_idx)):
            closest = np.argmin(np.abs(fixed_trials - t))
            self._f_offset[t] = t - fixed_trials[closest]

        # find trial start and end times
        self._trial_info = pd.DataFrame({
            'idx': self._trial_idx,
            'f_offset': self._f_offset,
            'RewardX': reward_xy[0],
            'RewardY': reward_xy[1],
            'StartTime': np.insert(reward_times[:-1], 0, self.start_time),
            'EndTime': reward_times, 
            'TrialN': ((self._f_offset + 3) % 4).astype(int) + 1
        })
        self._trial_info['Duration'] = self._trial_info.EndTime - self._trial_info.StartTime

    @property
    def trial_info(self):
        if self._trial_info is None:
            self.task_epoch()
        return self._trial_info

    @property
    def f_offset(self):
        if self._trial_info is None:
            self.task_epoch()
        return self._trial_info.f_offset

    @property
    def fixed_xy(self):
        if self._trial_info is None:
            self.task_epoch()
        return self._fixed_xy

    @property
    def reward_xy(self):
        if self._trial_info is None:
            self.task_epoch()
        return self._trial_info[['RewardX', 'RewardY']]

    @property
    def trial_starts(self):
        if self._trial_info is None:
            self.task_epoch()
        return self._trial_info['StartTime']

    @property
    def trial_ends(self):
        if self._trial_info is None:
            self.task_epoch()
        return self._trial_info['EndTime']

    @property
    def reward_times(self):
        return self.trial_ends

    @property
    def n_trials(self):
        if self._trial_info is None:
            self.task_epoch()
        return len(self._trial_info)

    @property
    def clu(self):
        return self._spk['clu']



if __name__ == '__main__':
    from replay_analysis.core.tint import TrialNPXAll
    import matplotlib.pyplot as plt

    DATA_ROOT = '/home/jgeerts/Data/Mattias'
    data_dir = os.path.join(DATA_ROOT, 'D', 'Day1')
    fn = os.listdir(op.join(DATA_ROOT, 'D', 'Day1'))[0]

    trial = TrialNPXAll(op.join(data_dir, fn))

    rm, _ = trial.get_spa_ratemap(c=5, bin_size_cm=1)
    plt.imshow(rm)
    plt.show()
