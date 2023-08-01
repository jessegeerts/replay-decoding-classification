import numpy as np
import matplotlib.pyplot as plt
import os

from definitions import load_experiment_info_freya, analysis_params
from core.tint import get_trial
from decoding_utils import get_spikes_raster
from linearized_track import get_z_track_distance
from replay_trajectory_classification import (
    SortedSpikesClassifier, Environment, RandomWalk,
    Uniform, Identity, estimate_movement_var)
from plotting import plot_classification_during_movement, plot_classification_during_replay
from event_detection import find_replay_events, find_replay_spikes


state_names = ['continuous', 'fragmented', 'stationary']
STATE_COLORS = {i: j for i, j in zip(state_names, ['r', 'g', 'b'])}

write_loc = os.path.join(os.getcwd(), 'results')
if not os.path.exists(write_loc):
    os.makedirs(write_loc)


# Load experiment data
# --------------------------------------------------------------------------------------------------------------
exp_info = load_experiment_info_freya()
index = 0

trial = get_trial(exp_info.loc[index].Animal, y=exp_info.loc[index].Year, m=exp_info.loc[index].Month,
                  d=exp_info.loc[index].Day, t='track1')
sampling_frequency = trial.pos_samp_rate
available_tets = np.array(trial.get_available_tets())
tetrodes = available_tets[available_tets < 8 if exp_info.loc[index].Animal == 'r2142' else available_tets >= 8]

save_loc = os.path.join(write_loc, exp_info.loc[index].Animal, 'track1')
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

position = trial.xy

# get data in right format
# --------------------------------------------------------------------------------------------------------------
spikes, time, cell_has_spikes = get_spikes_raster(trial)

print('Linearizing position...')
print('-' * 80)

# linearize the position data
# --------------------------------------------------------------------------------------------------------------
position_df, graph = get_z_track_distance(trial)
linearized_position = position_df.linear_position

plt.figure(figsize=(9, 5))

# train classifier
# --------------------------------------------------------------------------------------------------------------
print('Training classifier...')
print('-' * 80)
state_names = ['continuous', 'fragmented', 'stationary']


time_ind = slice(None)

movement_var = estimate_movement_var(linearized_position, sampling_frequency)

environment = Environment(place_bin_size=np.sqrt(movement_var))
continuous_transition_types = [[RandomWalk(movement_var=movement_var),  Uniform(), Identity()],
                                [Uniform(),                                   Uniform(), Uniform()],
                                [RandomWalk(movement_var=movement_var), Uniform(), Identity()],]
classifier = SortedSpikesClassifier(
    environments=[environment],
    continuous_transition_types=continuous_transition_types,
    sorted_spikes_algorithm='spiking_likelihood_kde',
    sorted_spikes_algorithm_params={'position_std': 3.0,
                                   }
)

classifier.fit(linearized_position[time_ind], spikes[time_ind, :])


# plot results
# --------------------------------------------------------------------------------------------------------------
print('Plotting results from movement time...')
print('-' * 80)
time_ind = slice(42100, 43500)
results_during_actual_movement = classifier.predict(spikes[time_ind], time=time[time_ind], state_names=state_names)

plot_classification_during_movement(time[time_ind], spikes[time_ind],
                                    results_during_actual_movement,
                                    position_df.linear_position[time_ind].to_numpy())
plt.show()

# find replay events
# --------------------------------------------------------------------------------------------------------------
print('Finding replay events...')
print('-' * 80)

replay_events = find_replay_events(trial, tetrodes, analysis_params,
                                   gauss_filter_std=analysis_params['GaussFiltStd'], method='MUA')
replay_event_spikes = find_replay_spikes(trial, replay_events, analysis_params, tetrodes)

replay_events['SpikeTimes'] = [val['times'] for val in replay_event_spikes.values()]
replay_events['CellIDs'] = [val['cell_id'] for val in replay_event_spikes.values()]
replay_events['UniqueCells'] = replay_events['CellIDs'].apply(np.unique)
replay_events['NumCells'] = replay_events['UniqueCells'].apply(len)
replay_events['NumSpikes'] = replay_events['SpikeTimes'].apply(len)


# classify replay events
# --------------------------------------------------------------------------------------------------------------
print('Classifying replay event...')
print('-' * 80)

# take example replay event (here the one with the most spikes)
event_id = replay_events['NumSpikes'].argmax()
event = replay_events.loc[event_id]

# get spikes raster for input into classifier
n_cells = spikes.shape[-1]
replay_time = event.TimeRange
replay_spikes = np.zeros((len(replay_time)-1, n_cells))
for cell_id in range(n_cells):
    spike_times = event.SpikeTimes[event.CellIDs == cell_id]
    replay_spikes[:, cell_id] = np.histogram(spike_times, replay_time)[0]


# classify replay event
results_during_replay = classifier.predict(replay_spikes, time=replay_time[:-1], state_names=state_names)

plot_classification_during_replay(replay_time, replay_spikes, results_during_replay)
plt.show()