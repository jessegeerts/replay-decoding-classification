import numpy as np
from definitions import analysis_params


def get_spikes_raster(trial):
    """From TrialAxona instance, get spike raster for inputting into decoding functions.
    """
    all_tet_times = []
    cell_ids = []
    for tetrode in trial.get_available_tets():
        for c in trial.get_available_cells(tetrode):
            spktm = trial.spk_times(tetrode, c, as_type='p')
            speed_at_spike = trial.speed[spktm]
            all_tet_times.append(spktm[speed_at_spike > analysis_params['sThresh']])
            cell_ids.append((tetrode, c))
    n_cells = len(all_tet_times)
    spikes_raster = np.zeros((trial.xy.shape[-1], n_cells))
    for c in range(n_cells):
        spikes_raster[all_tet_times[c], c] = 1
    has_spikes = [np.any(spikes_raster[:, c]) for c in range(spikes_raster.shape[1])]
    time_range = np.arange(0, trial.duration, 1 / trial.pos_samp_rate)
    return spikes_raster[:, has_spikes], time_range, has_spikes


def get_median_err_statespace(results, trial_data, time_index, true_position):
    dec_pos_id = results.acausal_posterior.argmax(axis=1)
    dec_pos = results.acausal_posterior['position'][dec_pos_id]
    abs_err = np.abs(
        true_position[time_index][trial_data.speed[time_index] >= analysis_params['sThresh']] - dec_pos[
            trial_data.speed[time_index] >= analysis_params['sThresh']])
    return np.median(abs_err)
