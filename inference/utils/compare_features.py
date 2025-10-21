import numpy as np
import efel

def _reduce_feature_value(value):
    if value is None:
        return np.nan
    arr = np.asarray(value, dtype=float)
    return float(np.nanmean(arr)) if arr.size > 0 else np.nan

def compare_experiment_and_noble_features(traces, experimental_trace, spiking_features, non_spiking_features):
    # Compute spikecounts for NOBLE traces
    sc_results  = efel.get_feature_values(traces, ["Spikecount"]) if len(traces) > 0 else []
    spikecounts = np.asarray([_reduce_feature_value(r.get("Spikecount", None)) for r in sc_results], dtype=float) if len(sc_results) > 0 else np.asarray([])

    # Experimental spiking
    exp_sc_res              = efel.get_feature_values([experimental_trace], ['Spikecount'])
    experimental_spikecount = _reduce_feature_value(exp_sc_res[0].get('Spikecount', None))
    experimental_is_spiking = (not np.isnan(experimental_spikecount)) and (experimental_spikecount > 0.0)

    # Experimental features
    if experimental_is_spiking:
        selected_feature_names = [f for f in spiking_features if f != 'Spikecount']
        experimental_features  = efel.get_feature_values([experimental_trace], selected_feature_names)[0]
    else:
        selected_feature_names = list(non_spiking_features)
        experimental_features  = efel.get_feature_values([experimental_trace], selected_feature_names)[0]

    # Select NOBLE traces by spiking status
    if experimental_is_spiking:
        selected_indices = [i for i, sc in enumerate(spikecounts) if not np.isnan(sc) and sc > 0.0]
        excluded_count   = int(np.sum(np.isnan(spikecounts) | (spikecounts <= 0.0)))
    else:
        selected_indices = [i for i, sc in enumerate(spikecounts) if np.isnan(sc) or sc <= 0.0]
        excluded_count   = int(np.sum(~np.isnan(spikecounts) & (spikecounts > 0.0)))

    selected_traces = [traces[i] for i in selected_indices]

    # Compute NOBLE features for selected traces
    noble_results = efel.get_feature_values(selected_traces, selected_feature_names) if len(selected_traces) > 0 else []

    # Reduce per trace and aggregate
    noble_feature_values = {name: [] for name in selected_feature_names}
    for res in noble_results:
        for name in selected_feature_names:
            noble_feature_values[name].append(_reduce_feature_value(res.get(name, None)))

    noble_median = {}
    noble_q1 = {}
    noble_q3 = {}
    for name in selected_feature_names:
        vals = np.asarray(noble_feature_values[name], dtype=float)
        noble_median[name] = np.nanmedian(vals) if vals.size > 0 else np.nan
        noble_q1[name]     = np.nanpercentile(vals, 25) if vals.size > 0 else np.nan
        noble_q3[name]     = np.nanpercentile(vals, 75) if vals.size > 0 else np.nan

    experimental_values = {name: _reduce_feature_value(experimental_features.get(name, None)) for name in selected_feature_names}
    selection_counts    = {"used": len(selected_indices), "excluded": excluded_count}

    return selected_feature_names, experimental_values, noble_median, noble_q1, noble_q3, selection_counts


