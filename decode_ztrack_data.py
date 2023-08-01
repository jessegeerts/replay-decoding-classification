import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from replay_trajectory_classification import SortedSpikesDecoder
from replay_trajectory_classification.continuous_state_transitions import estimate_movement_var, RandomWalk

from definitions import analysis_params
from definitions import load_experiment_info_freya
from core.tint import get_trial
from linearized_track import get_z_track_distance
from plotting import plot_statespace_decoding
from decoding_utils import get_spikes_raster, get_median_err_statespace


write_loc = os.path.join(os.getcwd(), 'results')
if not os.path.exists(write_loc):
    os.makedirs(write_loc)

print('-' * 80)
print('Loading data...')
print('-' * 80)
# Load experiment data
# --------------------------------------------------------------------------------------------------------------
exp_info = load_experiment_info_freya()
index = 0

trial = get_trial(exp_info.loc[index].Animal, y=exp_info.loc[index].Year, m=exp_info.loc[index].Month,
                  d=exp_info.loc[index].Day, t='track1')

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

# plot linearized position
# --------------------------------------------------------------------------------------------------------------
sampling_frequency = trial.pos_samp_rate
time = np.arange(position_df.linear_position.size) / sampling_frequency
plt.scatter(time, position_df.linear_position, clip_on=False, s=1, color='magenta')

plt.xlabel('Time')
plt.ylabel('Linear Position')
sns.despine()

# plot projected x and y position
# --------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1,2, figsize=(8,5))
plt.sca(ax[0])
plt.plot(time, position[0])
plt.plot(time, position_df['projected_x_position'])
plt.xlim(0,500)

plt.sca(ax[1])
plt.plot(time, position[1])
plt.plot(time, position_df['projected_y_position'])
plt.xlim(0,500)

ax[0].set_title('True and recovered X position')
ax[1].set_title('True and recovered Y position')
sns.despine(offset=5)
plt.show()


# implement decoding
# --------------------------------------------------------------------------------------------------------------
# only use data when the animal is running
lp = linearized_position[trial.speed >= analysis_params['sThresh']]
movement_spikes = spikes[trial.speed >= analysis_params['sThresh'], :]

movement_var = estimate_movement_var(linearized_position, sampling_frequency)

rw = RandomWalk(movement_var=movement_var)
decoder = SortedSpikesDecoder(transition_type=rw)

print('Fitting decoder...')
print('-' * 80)

decoder.fit(linearized_position, spikes)

# plot the estimated rate maps from the decoder
# --------------------------------------------------------------------------------------------------------------
g = (decoder.place_fields_ * sampling_frequency).plot(
        x="position", col="neuron", col_wrap=5, color="red", linewidth=2, alpha=0.9, zorder=1, label="Predicted")
g.axes[0, 0].set_ylabel("Firing Rate [spikes/s]")

plt.legend()
plt.show()

# plot transition matrix
# --------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

edge1, edge2 = np.meshgrid(decoder.environment.place_bin_edges_, decoder.environment.place_bin_edges_)
ax.pcolormesh(edge1, edge2, decoder.state_transition_.T, vmin=0.0, vmax=np.percentile(decoder.state_transition_, 99.9))
ax.set_title("Random Walk State Transition Matrix")
ax.set_ylabel("Position at time t-1")
ax.set_xlabel("Position at time t")
ax.axis("square")

# predict the animal's position
# --------------------------------------------------------------------------------------------------------------
time_ind = slice(1000, 60000)

print('Predicting position...')
results_d = decoder.predict(spikes[time_ind], time[time_ind])
plot_statespace_decoding(position_df.linear_position, results_d, spikes, time, time_ind)
print('-' * 80)

# compute median error
# --------------------------------------------------------------------------------------------------------------
get_median_err_statespace(results_d, trial, time_ind, position_df.linear_position)

plt.show()

print('Done!')
print('-' * 80)
