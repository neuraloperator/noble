import numpy as np
import efel
from efel import get_feature_values, get_mean_feature_values
from collections import defaultdict
from typing import Tuple
import torch


def extract_features(stimulus: np.ndarray, response: np.ndarray, data_config: dict) -> dict:
    """
    Extract electrophysiological features from neural response data using eFEL.
    
    This function processes neural response traces to extract quantitative features
    that characterize the neuron's firing behavior and membrane properties using
    the eFEL library. It handles both spiking and non-spiking responses.
    
    Args:
        stimulus (np.ndarray): Stimulus timeseries as a 1D numpy array. Should be the same length as response.
        response (np.ndarray): Neural response data as a 1D numpy array containing voltage
            traces in volts.
        data_config (dict): Configuration dictionary containing data generation parameters:
            - dt (float): Original time step in ms for generating the data pre-downsampling
            - ds_factor (int): Downsampling factor used in generating the training data
            - signal_length (float): Duration of stimulus and response signals in ms
    
    Returns:
        dict: Dictionary containing extracted electrophysiological features. Each feature
            is a list of values (one per trace), with None for features that could not
            be computed. Feature names include:
            - For spiking responses: AP1_peak, AP1_width, Spikecount, AHP_depth, etc.
            - For non-spiking responses: decay_time_constant_after_stim, sag_amplitude,
              steady_state_voltage, voltage_base, etc.
    """

    traces = []
    trace = {}


    signal_length = data_config['signal_length']
    dt            = data_config['dt']
    ds_factor     = data_config['ds_factor']

    time = np.arange(0, signal_length, dt)[::ds_factor]

    # Find stimulus start (first non-zero value)
    stim_start_idx = np.where(stimulus != 0)[0][0]
    
    # Find stimulus end (last non-zero value)
    stim_end_idx = np.where(stimulus != 0)[0][-1]

    trace['T'] = time

    trace['V'] = response * 1000  # V -> mV

    stim_start = trace['T'][stim_start_idx]
    stim_end   = trace['T'][stim_end_idx]

    trace['stim_start'] = [stim_start]
    trace['stim_end']   = [stim_end]

    traces.append(trace.copy())

    spikes     = efel.get_mean_feature_values(traces, ['Spikecount'])
    spikecount = spikes[0]['Spikecount']

    if spikecount == 0:
        feature_names = ['decay_time_constant_after_stim', 'sag_amplitude', 'steady_state_voltage',
                         'steady_state_voltage_stimend', 'voltage_base']
    else:
        feature_names = ['AHP1_depth_from_peak', 'AHP_depth', 'AHP_time_from_peak', 'AP1_peak',
                         'AP1_width', 'Spikecount', 'decay_time_constant_after_stim', 'depol_block',
                         'inv_first_ISI', 'mean_AP_amplitude', 'steady_state_voltage',
                         'steady_state_voltage_stimend', 'time_to_first_spike', 'voltage_base']

    features = efel.get_feature_values(traces, feature_names)
    return features

def update_feature_errors(input_batch: np.ndarray, 
                          gt_batch: np.ndarray, 
                          pred_batch: np.ndarray, 
                          data_config: dict,
                          epsilon: float = 1e-6) -> Tuple[dict, int, int]:
    """
    Update feature errors by comparing ground truth and predicted signals.
    
    This function extracts electrophysiological features from both ground truth and predicted
    signals, computes relative errors for each feature, and tracks missed firing/non-firing
    predictions.
    
    Args:
        input_batch (np.ndarray): Input stimulus batch of shape (batch_size, sequence_length)
        gt_batch (np.ndarray): Ground truth signal batch of shape (batch_size, sequence_length)
        pred_batch (np.ndarray): Predicted signal batch of shape (batch_size, sequence_length)
        data_config (dict): Configuration dictionary containing data generation parameters:
            - dt (float): Original time step in ms for generating the data pre-downsampling
            - ds_factor (int): Downsampling factor used in generating the training data
            - signal_length (float): Duration of stimulus and response signals in ms
        epsilon (float, optional): Small value to prevent division by zero in relative error
            computation. Defaults to 1e-6.
    
    Returns:
        Tuple[dict, int, int]: A tuple containing:
            - Dictionary mapping feature names to lists of relative errors
            - Count of missed firing predictions (true_spikes > 0, pred_spikes = 0)
            - Count of missed non-firing predictions (true_spikes = 0, pred_spikes > 0)
    """

    def safe_mean(val):
        if isinstance(val, (list, np.ndarray)):
            if len(val) == 0:
                return None
            return float(np.mean(val))
        return val
    
    errors = defaultdict(list)
    missed_firing = 0
    missed_non_firing = 0

    for stimulus, true_signal, pred_signal in zip(input_batch, gt_batch, pred_batch):
        true_feats = extract_features(stimulus, true_signal, data_config)[0]
        pred_feats = extract_features(stimulus, pred_signal, data_config)[0]

        # Spike count handling
        true_spikes = true_feats.get("Spikecount", 0)
        pred_spikes = pred_feats.get("Spikecount", 0)

        if true_spikes > 0 and pred_spikes == 0:
            missed_firing += 1
            continue
        elif true_spikes == 0 and pred_spikes > 0:
            missed_non_firing += 1
            continue

        # Feature-wise relative errors
        for feat in true_feats:
            true_val = safe_mean(true_feats[feat])
            pred_val = safe_mean(pred_feats.get(feat))

            if true_val is None or pred_val is None:
                continue

            rel_err = np.abs((pred_val - true_val)/(true_val + epsilon))

            errors[feat].append(rel_err)

    return errors, missed_firing, missed_non_firing
