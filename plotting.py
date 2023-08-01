import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


state_names = ['continuous', 'fragmented', 'stationary']

STATE_COLORS = {
    'stationary': '#9f043a',
    'fragmented': '#ff6944',
    'continuous': '#521b65',
    'stationary-continuous-mix': '#61c5e6',
    'fragmented-continuous-mix': '#2a586a',
    '': '#c7c7c7',
}


def plot_statespace_decoding(linearized_position, results, spikes_raster, time_range, time_index):
    cmap = plt.get_cmap('tab20')

    fig, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(9, 7))
    spike_ind, neuron_ind = np.nonzero(spikes_raster[time_index])
    c = [cmap.colors[ind % cmap.N] for ind in neuron_ind]
    axes[0].scatter(time_range[spike_ind], neuron_ind + 1, c=c, s=0.5, clip_on=False)
    axes[0].set_yticks((1, spikes_raster.shape[1]))
    axes[0].set_ylabel('Cells')
    results.causal_posterior.plot(x="time", y="position", ax=axes[1], cmap="bone_r", vmin=0.0, vmax=0.05,
                                  clip_on=False)
    axes[1].plot(time_range[time_index], linearized_position[time_index], color="magenta", linestyle="--", linewidth=3,
                 clip_on=False)
    axes[1].set_xlabel("")
    results.acausal_posterior.plot(x="time", y="position", ax=axes[2], cmap="bone_r", vmin=0.0, vmax=0.05,
                                   clip_on=False)
    axes[2].plot(time_range[time_index], linearized_position[time_index], color="magenta", linestyle="--", linewidth=3,
                 clip_on=False)
    axes[2].set_xlabel('Time [s]')
    sns.despine(offset=5)
    return fig, axes


def plot_classification(replay_time, test_spikes, results):
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
    spike_time_ind, neuron_ind = np.nonzero(test_spikes)
    axes[0].scatter(replay_time[spike_time_ind], neuron_ind, color='black', zorder=1,
                    marker='|', s=80, linewidth=3)
    axes[0].set_yticks((0, test_spikes.shape[1]))
    axes[0].set_ylabel('Neuron Index')
    replay_probability = results.causal_posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        axes[1].plot(prob.time, prob.values, linewidth=4, label=state, color=STATE_COLORS[state])
    axes[1].set_ylabel('Probability')
    axes[1].set_yticks([0, 1])
    axes[1].set_ylim((-0.01, 1.05))
    axes[1].legend(bbox_to_anchor=(1.15, 0.95), loc='upper right', fancybox=False, shadow=False,
                   ncol=1, frameon=False)

    results.causal_posterior.sum('state').plot(
        x='time', y='position', robust=True, vmin=0.0, ax=axes[2])
    axes[2].set_ylabel('Position [cm]')
    plt.xlim((replay_time.min(), replay_time.max()))
    axes[-1].set_xlabel('Time [ms]')
    sns.despine()


def plot_classification_during_movement(time_range, spike_array, classifier_results, position_data):
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.sca(ax[0])
    spike_time_ind, neuron_ind = np.nonzero(spike_array)
    ax[0].scatter(time_range[spike_time_ind], neuron_ind, color='black', zorder=1,
                  marker='|', s=20, linewidth=1)
    ax[0].set_ylabel('Neuron ID')
    plt.sca(ax[1])
    latent_position_posterior = np.array(classifier_results.acausal_posterior.sum('state'))
    max_a_posteriori = np.argmax(latent_position_posterior, axis=1)
    decoded_position = classifier_results.acausal_posterior['position'][max_a_posteriori]
    plt.plot(time_range, decoded_position)
    plt.plot(time_range, position_data)
    plt.legend(['Decoded', 'Real'])
    ax[1].set_ylabel('Position (cm)')
    plt.title('Decoding spikes during actual movement')
    replay_probability = classifier_results.causal_posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        ax[2].plot(prob.time, prob.values, linewidth=2, label=state, color=STATE_COLORS[state])
    ax[2].set_ylabel('Probability')
    ax[2].set_yticks([0, 1])
    ax[2].set_ylim((-0.01, 1.05))
    ax[2].legend(bbox_to_anchor=(1.15, 0.95), loc='upper right', fancybox=False, shadow=False,
                 ncol=1, frameon=False)
    ax[2].set_xlabel('Time (s)')


def plot_classification_during_replay(replay_time, test_spikes, results):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
    spike_time_ind, neuron_ind = np.nonzero(test_spikes)
    axes[0].scatter(replay_time[spike_time_ind], neuron_ind, color='black', zorder=1,
                    marker='|', s=80, linewidth=3)
    axes[0].set_yticks((0, test_spikes.shape[1]))
    axes[0].set_ylabel('Neuron Index')
    replay_probability = results.acausal_posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        axes[1].plot(prob.time, prob.values, linewidth=4, label=state, color=STATE_COLORS[state])
    axes[1].set_ylabel('Probability')
    axes[1].set_yticks([0, 1])
    axes[1].set_ylim((-0.01, 1.05))
    axes[1].legend(bbox_to_anchor=(1.15, 0.95), loc='upper right', fancybox=False, shadow=False,
                   ncol=1, frameon=False)

    results.acausal_posterior.sum('state').plot(
        x='time', y='position', robust=True, vmin=0.0, ax=axes[2])
    axes[2].set_ylabel('Position [cm]')
    plt.xlim((replay_time.min(), replay_time.max()))
    axes[-1].set_xlabel('Time [ms]')
    sns.despine()
