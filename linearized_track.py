import numpy as np
import scipy as sp
from numpy_groupies import aggregate
from utils.custom_np import argsort_inverse, ignore_divide_by_zero
from replay_trajectory_classification import make_track_graph
from track_linearization import get_linearized_position


def get_spa_ratemap_1d(trial, x, t=None, c=None, bin_size_cm=2.5, smoothing_bins=5,
                       w=None, spk_pos_idx=None, spk_weights=None,
                       smoothing_type='boxcar', norm=None,
                       nodwell_mode='ma', spike_shifts_s=None):
    """for spikes on tetrode t, cell c, this returns a spatial ratemap, using
    the specified bin size in cm and smoothing type and width.

    `norm` - ``None`` means give rate in Hz, ``pos`` means rate in
     Hz times by pos_samp_rate, i.e. rate per pos sample; ``counts`` means use raw spike
     counts (and ignore dwell time entierly). Alternatively `'max'` to divide
     by peak  rate or `'mean'` to divide by the cell's average firing rate throughout
     the trial.  Or an integer on the interval `(0,100]` to use the given
     percentile in the (valid part of the) ratemap as the normalizer (e.g.
     50 means normalize to median. Or `'rank'` in which case each rate is
     replaced by its percentile rank in the list of valid bins.

    ``spk_weights``, if not None should be the same length as the number of spikes.
    It will be used in the accumulation instead of 1, i.e. each spike can be given
    a specific weighting rather than just being counted.

    In addition of the usual ``boxcar``,  ``gaussian``, and ``None``,
    ``smoothing_type`` can be ``adaptive``, in which case the ratemap request
    is passed on to ``get_ratemap_adaptive``, with  ``sigma`` set to
    ``smoothing_bins`` (try using sigma=200).

    If ``t`` and ``c`` are not ``None``, it returns a ratemap, with bins in Hz.
    Otherwise it returns a dwell map with bins in seconds. If ``as_count`` is
    ``True`` and ``t`` and ``c`` are integers, it returns a ratemap with the bins
    giving spike counts per bin, i.e. the division by the dwell map is not done.

    If ``spike_shifts_s`` is not ``None``, it is an array of temporal
    shifts to apply to the spike train, computing a separate ratemap for
    each shift, ie. the result is a stack of ratemaps not a single ratemp.
    Support for this option may have limitations.
    """

    bin_size_cm = float(bin_size_cm)

    # prepare xy, w, h, and spk_pos_idx (if not explicitly provided)
    w = np.ceil(x.max()) + 1 if w is None else w

    if spk_pos_idx is not None:
        pass
    elif t is not None:
        spk_pos_idx = trial.spk_pos_idx(t=t, c=c, as_type='p')
    elif c is not None:
        raise ValueError("if you provide cell number you must also give tet number.")

    if smoothing_type == 'boxcar' and smoothing_bins:
        filter_func = lambda x: sp.ndimage.uniform_filter1d(x, smoothing_bins, mode='constant')
    elif smoothing_type == 'gaussian' and smoothing_bins:
        filter_func = lambda x: sp.ndimage.gaussian_filter1d(x, smoothing_bins, mode='constant')
    elif smoothing_type is None or not smoothing_bins:
        filter_func = lambda x: x
    else:
        raise ValueError("unknown smoothing type: {}".format(smoothing_type))

    xy_idx, w, h = trial.xy_bin_inds(bin_size_cm, xy=x, w=w, h=1)

    # prepare dwell and/or nodwell
    dwell = aggregate(xy_idx[:], 1., size=w)
    nodwell = dwell == 0
    if norm == 'counts':
        pass  # don't care about dwell any more (only nodwell was needed)
    else:
        dwell = filter_func(dwell)
        if norm is None:
            dwell /= trial.pos_samp_rate  # now in seconds, otherwise relative norm

    if spk_pos_idx is None:
        rm = dwell
    else:
        # prepare spikes and rate, and normalise rate
        if spike_shifts_s is None:
            spk = aggregate(xy_idx[spk_pos_idx], spk_weights or 1., size=w)
        else:
            # deal with shuffles...
            if spk_weights is not None:
                raise NotImplementedError("spk_weights not supported for shifted spikes.")
            spike_shifts_pos = (spike_shifts_s * trial.pos_samp_rate).astype(
                int)  # this is not perfect, but good enough
            spike_shifts_pos.shape = 1, -1
            n_shifts = spike_shifts_pos.size
            n_spikes = spk_pos_idx.size
            spk_pos_idx = spk_pos_idx[:, np.newaxis] + spike_shifts_pos
            spk_pos_idx %= int(trial.n_pos)
            plane_idx = np.repeat(np.arange(n_shifts)[np.newaxis, :], n_spikes, axis=0)
            spk = aggregate([x.ravel() for x in
                             tuple(xy_idx[::-1, spk_pos_idx]) + (plane_idx,)],
                            1., size=[h, w, n_shifts])
            dwell = dwell[..., np.newaxis]
            nodwell = np.repeat(nodwell[..., np.newaxis], n_shifts, axis=-1)

        spk = filter_func(spk)
        if norm == 'counts':
            rm = spk
        else:
            with ignore_divide_by_zero():
                rm = spk / dwell
            if norm is None:
                pass  # true Hz
            elif norm == "pos":
                pass  # rate in per-pos-sample units
            elif norm == 'max':
                rm *= 1. / np.nanmax(rm)
            elif norm == 'mean':
                rm *= trial.duration / len(spk_pos_idx)
            elif norm == 'rank':
                mask = ~np.isnan(rm)
                rm[mask] = argsort_inverse(rm[mask]) * (100.0 / np.sum(mask))
            elif 0 < norm <= 100:
                rm *= 1.0 / np.nanpercentile(rm, norm)  # eg. norm by median
            else:
                raise ValueError("unknown norm type: {}".format(norm))

    # apply nodwell_mode
    if nodwell_mode == 'nan':
        rm[nodwell] = np.nan
    elif nodwell_mode == 'ma':
        rm = np.ma.array(rm, mask=nodwell)
    elif np.isscalar(nodwell_mode) and isinstance(nodwell_mode, (float, int)):
        rm[nodwell] = nodwell_mode
    elif isinstance(nodwell_mode, str) and nodwell_mode.startswith('interp'):
        # this is very basic, nearest-neighbor interp, with no averaging for ties.
        nearest_idx0, nearest_idx1 = sp.ndimage.distance_transform_edt(
            nodwell, return_indices=True, return_distances=False)
        rm[nodwell] = rm[nearest_idx0[nodwell], nearest_idx1[nodwell]]
    else:
        raise Exception("unrecognised nodwell mode")

    return rm, bin_size_cm


def get_1d_rate_maps_for_decoding(trial, x, tetrodes, params):
    """Get all ratemaps for all cells, filtered for running speed.

    Spikes fired when the animal's running speed is below a threshold (params['sThresh'])

    Args:
        trial (TrialAxonaAll): trial object.
        tetrodes (iterable): List of tetrodes to use.
        params (dict): Contains speed threshold 'sThresh' and 'binSize' keys.

    Returns: [n_cells, n_position_bins] array with firing rate maps

    """
    rate_maps = []
    for tet in tetrodes:
        try:
            available_cells = trial.get_available_cells(tet)
        except KeyError:
            available_cells = []
        #
        for c in available_cells:
            spk_pos_ind = trial.spk_times(t=tet, c=c, as_type='p')
            speed_at_spike = trial.speed[spk_pos_ind]
            spk_pos_ind = spk_pos_ind[speed_at_spike > params['sThresh']]
            rm = get_spa_ratemap_1d(trial, x, t=tet, c=c, spk_pos_idx=spk_pos_ind,
                                    bin_size_cm=params['binSize'], smoothing_bins=7)[0]
            minx, maxx = (int(x.min()), int(x.max()))
            rm = rm[minx: maxx]
            rate_maps.append(rm)
    return np.array(rate_maps)


def get_z_track_distance(trial_data, return_graph=True):
    """
    TODO: optimize this so there's no gaps
    Args:
        trial_data:

    Returns:

    """

    margin = 15.
    minx = trial_data.xy[0][5000:-5000].min()
    maxx = trial_data.xy[0][5000:-5000].max()
    miny = trial_data.xy[1][5000:-5000].min()
    maxy = trial_data.xy[1][5000:-5000].max()
    node_positions = [(minx + margin, maxy - margin),  # xy position of node 0
                      (maxx - margin, maxy - margin),  # xy position of node 1
                      (minx + margin, miny + margin),  # xy position of node 2
                      (maxx - margin, miny + margin),  # xy position of node 3
                      ]

    edges = [(0, 1),  # connects node 0 and node 1
             (1, 2),  # connects node 1 and node 2
             (2, 3),  # connects node 2 and node 3
             ]

    edge_order = [(3, 2),
                  (2, 1),  # node 2 to node 1
                  (1, 0),  # node 1 to node 0
                  ]
    edge_spacing = 0  # no spacing between edges
    track_graph = make_track_graph(node_positions, edges)
    position_df = get_linearized_position(trial_data.xy.T, track_graph, edge_order=edge_order,
                                          edge_spacing=edge_spacing, use_HMM=False)

    if return_graph:
        return position_df, track_graph
    else:
        return position_df
