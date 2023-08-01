import os
import platform
from datetime import datetime
import math

import pandas as pd
import seaborn as sns


COLOR_PALETTE = sns.color_palette("Set2")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# data locations
hostname = platform.node()
if hostname == 'tech-Precision-7820-Tower':  # ICN machine
    DATA_ROOT_GUIFEN = '/home/jgeerts/Data/Guifen'
    DATA_ROOT_FREYA = '/home/jgeerts/Data/Olafsdottir2016'
    DATA_ROOT_MATTIAS = '/home/jgeerts/Data/Mattias'
elif hostname == 'Jesses-MacBook-Pro.local':  # MacBook
    DATA_ROOT_GUIFEN = '/Users/jessegeerts/Data/Guifen'
else:
    DATA_ROOT_GUIFEN = '/home/jesse/Data/Guifen/'


analysis_params = {
    'binSize': 2.5,  # Bin size for firing rate maps (cm)
    'sThresh': 10,  # Speed threshold for firing rate maps (cm/s)
    'mov_tBin': .5,  # Time bin for decoding location during movement (s)
    'rippleBand': [150, 250],  # Ripple frequency band (Hz)
    'durThresh': 0.04,  # Replay event duration (s) NOTE NO UPPER DURATION
    'zThresh': 3,  # Replay peak threshold (Z)
    'vThresh': 10,  # Replay movement speed threshold (cm/s)
    'rThresh': 10,  # Firing rate threshold for cell inclusion (Hz)
    'rep_tBin': 0.01,  # Time bin for decoding location during replay (s)
    'makeNans': True,  # Set empty time bins to NaN during decoding?
    'cellThresh': 3,  # Threshold number of cells per event to decode
    'spkThresh': 5,  # Threshold number of spikes per event to decode
    'thetaBand': [7, 11],  # Theta frequency band (Hz)
    'nShuf': 1000,  # Number of shuffles to establish decoding error due to chance
    'minFieldSz': 20,  # Minimum field size (cm) ?
    'phaseRange': 4 * math.pi,  # Phase range
    'tacSpkThrsh': 20,  # ?
    'GaussFiltStd': 5.  # Size of the Gaussian filter used for event detection.
}


def load_experiment_info_guifen():
    trial_info_header = ['Animal', 'RecordLocs', 'ExperimentN', 'Trial1', 'BoxSize1', 'Trial2', 'VBoxSize2', 'Trial3',
                         'VBoxSize3', 'FixedRewardModulus']
    exp_info = pd.read_csv(os.path.join(DATA_ROOT_GUIFEN, 'trial_info.csv'),
                           header=None, names=trial_info_header).drop(8).sort_values('Animal')
    exp_info['E1inCA1'] = exp_info['RecordLocs'].apply(lambda x: True if x[0] == 'C' else False)
    exp_info['E2inCA1'] = exp_info['RecordLocs'].apply(lambda x: True if x[1] == 'C' else False)
    exp_info['Animal'] = exp_info['Animal'].astype('int').astype('str')
    return exp_info


def load_experiment_info_freya():
    animals = [a for a in os.listdir(DATA_ROOT_FREYA) if os.path.isdir(os.path.join(DATA_ROOT_FREYA, a))]
    sessions = {a: os.listdir(os.path.join(DATA_ROOT_FREYA, a)) for a in animals}
    fn = os.path.join(DATA_ROOT_FREYA, animals[0], sessions[animals[0]][0],
                      sessions[animals[0]][0].replace('-', '') + '_' + animals[0].capitalize() + '_' + 'sleepPOST')

    exp_info = {'Animal': [], 'DateTime': [], 'Year': [], 'Month': [], 'Day': []}
    for a, sess in sessions.items():
        for s in sess:
            time = datetime.strptime(s, '%Y-%m-%d')
            exp_info['Animal'].append(a)
            exp_info['DateTime'].append(time)
            exp_info['Year'].append(time.year)
            exp_info['Month'].append(time.month)
            exp_info['Day'].append(time.day)

    exp_info = pd.DataFrame.from_dict(exp_info)

    # TODO: add trial types
    return exp_info
