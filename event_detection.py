import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, binary_dilation, binary_erosion
from scipy.signal import firwin, filtfilt, hilbert
from skimage import measure

from definitions import RESULTS_DIR
from core.tint import TrialAxonaAll


def _find_replay_events_swr(trial, params, gauss_filter_std=5.):
    """Find replay events using the SWR method:

        1. Filter the data in the ripple band, smooth and z-score the output
        2. find connected regions above the mean, then throw out events with
           a peak value and duration lower than the corresponding thresholds,
           or with running speed above threshold.
        3. Connect events separated by less than the minimum event duration
        4. Finally, collate the details of each event in a DataFrame

    Args:
        trial (TrialAxonaAll): The Trial object.
        params (Dict) : Dictionary of user-defined parameters.
        gauss_filter_std: Standard deviation of Gaussian smoothing filter (ms)

    Returns: DataFrame containing info for each ripple event.
    """
    b = firwin(401, np.array(params['rippleBand']), fs=trial.egf_samp_rate, pass_zero='bandpass')
    filtered_eeg = filtfilt(b, 1., trial.egf(bad_as=None))
    smooth_amp = np.abs(hilbert(filtered_eeg))
    smooth_amp = gaussian_filter1d(smooth_amp,
                                   float(gauss_filter_std / 1000 * trial.egf_samp_rate))  # why divide by 1000?
    smooth_amp = (smooth_amp - smooth_amp.mean()) / smooth_amp.std()  # z-score

    swr_time = np.arange(len(smooth_amp)) / trial.egf_samp_rate
    event_cands, event_labels, speed = find_and_clean_connected_events(smooth_amp, swr_time, params, trial)
    event_labels, event_props = connect_nearby_events(event_cands, event_labels, params, trial.egf_samp_rate)

    # store event data in a DataFrame
    df = pd.DataFrame({})
    df['TimeStart'] = pd.Series([event_props[i - 1].coords[0, 0] for i in event_labels]) / trial.egf_samp_rate
    df['TimeEnd'] = pd.Series([event_props[i - 1].coords[-1, 0] for i in event_labels]) / trial.egf_samp_rate
    df['TimeRange'] = [event_props[i - 1].coords[:, 0] / trial.egf_samp_rate for i in event_labels]
    df['RawEEG'] = [trial.egf(bad_as=None)[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['FiltEEG'] = [filtered_eeg[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['zAmp'] = [smooth_amp[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['RunSpeed'] = [speed[event_props[i - 1].coords[:, 0]] for i in event_labels]
    return df


def find_replay_events_swr(lfp_signal, speed, params, lfp_samp_rate, pos_samp_rate, gauss_filter_std=5.):
    """Find replay events using the SWR method:

        1. Filter the data in the ripple band, smooth and z-score the output
        2. find connected regions above the mean, then throw out events with
           a peak value and duration lower than the corresponding thresholds,
           or with running speed above threshold.
        3. Connect events separated by less than the minimum event duration
        4. Finally, collate the details of each event in a DataFrame

    Args:
        trial (TrialAxonaAll): The Trial object.
        params (Dict) : Dictionary of user-defined parameters.
        gauss_filter_std: Standard deviation of Gaussian smoothing filter (ms)

    Returns: DataFrame containing info for each ripple event.
    """
    b = firwin(401, np.array(params['rippleBand']), fs=lfp_samp_rate, pass_zero='bandpass')
    filtered_eeg = filtfilt(b, 1., lfp_signal)
    smooth_amp = np.abs(hilbert(filtered_eeg))
    smooth_amp = gaussian_filter1d(smooth_amp,
                                   float(gauss_filter_std / 1000 * lfp_samp_rate))  # why divide by 1000?
    smooth_amp = (smooth_amp - smooth_amp.mean()) / smooth_amp.std()  # z-score

    swr_time = np.arange(len(smooth_amp)) / lfp_samp_rate
    event_cands, event_labels, speed = find_and_clean_connected_events2(smooth_amp, speed, swr_time, params, pos_samp_rate)
    event_labels, event_props = connect_nearby_events(event_cands, event_labels, params, lfp_samp_rate)

    # store event data in a DataFrame
    df = pd.DataFrame({})
    df['TimeStart'] = pd.Series([event_props[i - 1].coords[0, 0] for i in event_labels]) / lfp_samp_rate
    df['TimeEnd'] = pd.Series([event_props[i - 1].coords[-1, 0] for i in event_labels]) / lfp_samp_rate
    df['TimeRange'] = [event_props[i - 1].coords[:, 0] / lfp_samp_rate for i in event_labels]
    df['RawEEG'] = [lfp_signal[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['FiltEEG'] = [filtered_eeg[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['zAmp'] = [smooth_amp[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['RunSpeed'] = [speed[event_props[i - 1].coords[:, 0]] for i in event_labels]
    return df


def find_replay_events_mua(trial, tetrodes, params, gauss_filter_std=5., mua_bin_size=.001, add_filt_eeg=True):
    """Find replay events using the SWR method:

        1. Generate a multi unit activity histogram, smooth and z-score
        2. find connected regions above the mean, then throw out events with
           a peak value and duration lower than the corresponding thresholds,
           or with running speed above threshold.
        3. Connect events separated by less than the minimum event duration
        4. Finally, collate the details of each event in a DataFrame

    Args:
        trial (TrialAxonaAll): The Trial object.
        tetrodes (Union[list, array]): List of tetrodes to use.
        params (dict) : Dictionary of user-defined parameters.
        gauss_filter_std (float): Standard deviation of Gaussian smoothing filter (ms)
        mua_bin_size (float): Size of multi-unit activity histogram bins (s)
        add_filt_eeg (bool): Whether to add the filtered EEG to the output.

    Returns: DataFrame containing info for each MUA event.
    """
    # mu_time = np.arange(mua_bin_size, trial.duration + mua_bin_size, mua_bin_size)
    mu_time = np.arange(0, trial.duration + mua_bin_size, mua_bin_size)
    mua = np.zeros(mu_time.shape[0] - 1)
    cell_count = 0
    for tet in tetrodes:
        available_cells = trial.get_available_cells(tet)
        if len(available_cells) > 0:
            for c in available_cells:
                mean_rate = len(trial.spk_times(t=tet, c=c)) / trial.duration
                if mean_rate <= params['rThresh']:
                    spike_times = trial.spk_times(t=tet, c=c)
                    mua += np.histogram(spike_times, mu_time)[0]
                    cell_count += 1

    smooth_mua = gaussian_filter1d(mua, round(gauss_filter_std / 1000 / mua_bin_size))
    smooth_mua = (smooth_mua - smooth_mua.mean()) / smooth_mua.std()

    event_cands, event_labels, speed = find_and_clean_connected_events(smooth_mua, mu_time[:-1], params, trial)
    sample_rate = 1 / mua_bin_size
    event_labels, event_props = connect_nearby_events(event_cands, event_labels, params, sample_rate)

    # store event data in a DataFrame
    df = pd.DataFrame({})
    df['TimeStart'] = pd.Series([event_props[i - 1].coords[0, 0] for i in event_labels]) / sample_rate
    df['TimeEnd'] = pd.Series([event_props[i - 1].coords[-1, 0] for i in event_labels]) / sample_rate
    df['TimeRange'] = [event_props[i - 1].coords[:, 0] / sample_rate for i in event_labels]
    df['zAmp'] = [smooth_mua[event_props[i - 1].coords[:, 0]] for i in event_labels]
    df['RunSpeed'] = [speed[event_props[i - 1].coords[:, 0]] for i in event_labels]

    if add_filt_eeg and trial.has_egf:
        b = firwin(401, np.array(params['rippleBand']), fs=trial.egf_samp_rate, pass_zero='bandpass')
        filtered_eeg = filtfilt(b, 1., trial.egf(bad_as=None))
        df['FiltEEG'] = [filtered_eeg[event_props[i - 1].coords[:, 0]] for i in event_labels]
    return df


def connect_nearby_events(event_cands, event_labels, params, sample_rate):
    """Connect events separated by less than the minimum event duration.

    Args:
        event_cands:
        event_labels:
        params:
        sample_rate:

    Returns:

    """
    min_duration = int(params['durThresh'] * sample_rate)
    events = np.where(np.in1d(event_cands, event_labels), True, False)
    events = binary_dilation(events, np.ones(min_duration, dtype=bool))
    events = binary_erosion(events, np.ones(min_duration, dtype=bool))
    label = measure.label(events)
    event_labels = np.unique(label)[1:]
    event_props = measure.regionprops(label[:, np.newaxis])
    durations = np.array([e.area for e in event_props]) / sample_rate
    event_labels = event_labels[durations > params['durThresh']]
    return event_labels, event_props


def find_and_clean_connected_events(signal, time_bins, params, trial):
    """Find connected regions above the mean, then throw out events with
    a peak value and duration lower than the corresponding thresholds,
    or with running speed above threshold.

    Args:
        signal (np.ndarray):
        time_bins (np.ndarray):
        params (dict):
        trial (TrialAxonaAll):

    Returns:
    """
    event_cands = measure.label(signal >= 0)
    event_cand_labels = np.unique(event_cands)[1:]
    region_props = measure.regionprops(event_cands[:, np.newaxis], signal[:, np.newaxis])
    peak_z = np.array([t.max_intensity for t in region_props])
    f = interp1d(np.arange(len(trial.speed)) / trial.pos_samp_rate, trial.speed, bounds_error=False)
    speed = f(time_bins)
    region_props = measure.regionprops(event_cands[:, np.newaxis], speed[:, np.newaxis])
    peak_v = np.array([t.max_intensity for t in region_props])
    event_labels = event_cand_labels[np.logical_and(peak_z >= params['zThresh'], peak_v < params['vThresh'])]
    return event_cands, event_labels, speed


def find_and_clean_connected_events2(signal, speed, time_bins, params, pos_samp_rate):
    """Find connected regions above the mean, then throw out events with
    a peak value and duration lower than the corresponding thresholds,
    or with running speed above threshold.

    Args:
        signal (np.ndarray):
        time_bins (np.ndarray):
        params (dict):
        trial (TrialAxonaAll):

    Returns:
    """
    event_cands = measure.label(signal >= 0)
    event_cand_labels = np.unique(event_cands)[1:]
    region_props = measure.regionprops(event_cands[:, np.newaxis], signal[:, np.newaxis])
    peak_z = np.array([t.max_intensity for t in region_props])
    f = interp1d(np.arange(len(speed)) / pos_samp_rate, speed, bounds_error=False)
    speed = f(time_bins)
    region_props = measure.regionprops(event_cands[:, np.newaxis], speed[:, np.newaxis])
    peak_v = np.array([t.max_intensity for t in region_props])
    event_labels = event_cand_labels[np.logical_and(peak_z >= params['zThresh'], peak_v < params['vThresh'])]
    return event_cands, event_labels, speed


def find_replay_events(trial, tetrodes, params, method='SWR', gauss_filter_std=5.):
    """Find replay events using either the SWR or MUA method.

    Args:
        trial (TrialAxonaAll): The Trial object.
        tetrodes (Union[list, np.ndarray]): List of tetrodes to use.
        params (dict): Dictionary of user-defined parameters.
        method (str): Detection method: 'SWR' or 'MUA'.
        gauss_filter_std (Union[float, int]): Standard deviation of Gaussian smoothing filter (ms)

    Returns: DataFrame with information for each detected event.
    """
    if method == 'SWR':
        if trial.has_egf:
            result = find_replay_events_swr(trial, params, gauss_filter_std=gauss_filter_std)
        else:
            result = pd.DataFrame({})
    elif method == 'MUA':
        result = find_replay_events_mua(trial, tetrodes, params, gauss_filter_std=gauss_filter_std)
    else:
        raise ValueError('Not a known event detection method.')
    return result


def find_replay_spikes(trial, events, params, tetrodes):
    """Extract spiking activity during each replay event.

        1. Collate all spike times to single array
        2. cycle through each replay event, collate spiking data

    Args:
        trial (TrialAxonaAll): Trial object.
        events (pd.DataFrame): DataFrame holding the candidate event data (created by find_replay_events)
        params (dict): Dictionary of user-defined parameters.
        tetrodes (Union[list, np.ndarray]): List of tetrodes to use.

    Returns (dict): Returns a nested dictionary with, per event, the spike times and associated cell ids.

    """
    cell_count = 1
    spike_times = []
    for t in tetrodes:
        for c in trial.get_available_cells(t):
            mean_rate = len(trial.spk_times(t=t, c=c)) / trial.duration
            if mean_rate < params['rThresh']:
                st = trial.spk_times(t=t, c=c)
                spike_times.append(np.array([st, np.ones(len(st)) * cell_count]).T)
                cell_count += 1
    spike_times = np.concatenate(spike_times, axis=0)

    event_spikes = {}
    for e in range(len(events)):
        event_mask = np.logical_and(spike_times[:, 0] >= events.loc[e]['TimeStart'],
                                    spike_times[:, 0] <= events.loc[e]['TimeEnd'])
        event_spikes[e] = {'times': spike_times[event_mask][:, 0], 'cell_id': spike_times[event_mask][:, 1].astype('int')}
    return event_spikes


def get_event_data(animal, experiment, tetrode_set, detection_method='MUA'):
    data_path = os.path.join(RESULTS_DIR, animal, experiment, str(tetrode_set))
    return pd.read_pickle(os.path.join(data_path, 'replay_event_data_{}'.format(detection_method)))