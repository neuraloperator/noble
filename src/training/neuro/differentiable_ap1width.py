import numpy as np
import efel
from collections import defaultdict
from typing import Tuple
import torch

EPSILON = 1e-9


def find_dc_stimulus_onset_offset(stimulus: np.ndarray, dt: float, ds: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the onset and offset times of non-zero stimulus in a batch of time series.

    Args:
        stimulus (np.ndarray): Array of shape (batch_size, time_steps)
        dt (float): Time step in ms

    Returns:
        Tuple[np.ndarray, np.ndarray]: start_times, end_times arrays of shape (batch_size,)
    """
    stimulus = np.asarray(stimulus)

    if stimulus.ndim > 2:
        stimulus = np.squeeze(stimulus)

    batch_size, time_steps = stimulus.shape
    is_nonzero = stimulus != 0

    first_indices = np.argmax(is_nonzero, axis=1)
    last_indices  = time_steps - 1 - np.argmax(np.fliplr(is_nonzero), axis=1)

    all_zero_mask = ~np.any(is_nonzero, axis=1)
    first_indices = first_indices.astype(float)
    last_indices  = last_indices.astype(float)
    first_indices[all_zero_mask] = 0.0
    last_indices[all_zero_mask]  = time_steps - 1.0

    start_times = first_indices * dt * ds
    end_times   = last_indices * dt * ds

    return start_times, end_times

def compute_threshold_voltage_and_ap1_width(
    stimulus: torch.Tensor,
    voltage: torch.Tensor,
    time: torch.Tensor,
    dt: float = 0.02,
    ds: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the true threshold voltage and AP1 width for a batch.

    Args:
        stimulus (torch.Tensor): (batch_size, time_steps)
        voltage (torch.Tensor): (batch_size, time_steps, 1)
        time (torch.Tensor): (time_steps,)
        dt (float): Time step in ms

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: thresholds, ap1_widths
    """
    # print(f"Voltage shape: {voltage.shape}")
    batch_size, time_steps = voltage.shape
    device = voltage.device

    v_true_bt = voltage.squeeze(-1)  # (B, T)
    q10 = torch.quantile(v_true_bt, 0.10, dim=1)
    q90 = torch.quantile(v_true_bt, 0.90, dim=1)
    fallback_thr = 0.5 * (q10 + q90)  # (B,)

    voltage_np = voltage.cpu().detach().numpy()
    time_np = time.cpu().numpy()
    stimulus_np = stimulus.cpu().numpy()

    start_times, end_times = find_dc_stimulus_onset_offset(stimulus_np, dt, ds)

    # ORIGINAL PRINTS (kept)
    # print(f"Stimulus start times: {start_times}")
    # print(f"Stimulus end times: {end_times}")

    trials = []
    for i in range(batch_size):
        trials.append({
            'V': voltage_np[i],
            'T': time_np,
            'stim_start': [start_times[i]],
            'stim_end': [end_times[i]],
            'interp_step': [dt * ds],
        })

    features = efel.get_feature_values(trials, ['AP1_width', 'min_AHP_indices', 'peak_indices'])

    # Start from safe fallbacks; overwrite where EFEL is valid
    thresholds = fallback_thr.to(device=device, dtype=torch.float32).clone()
    min_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    peak_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    ap1_widths = torch.zeros((batch_size,), dtype=torch.float32, device=device)

    for i, feature in enumerate(features):
        if feature is None:
            continue
        width = feature.get('AP1_width')
        minA = feature.get('min_AHP_indices')
        peak = feature.get('peak_indices')
        if width is not None and len(width) > 0 and minA is not None and len(minA) > 0 and peak is not None and len(peak) > 0:
            min_indices[i]  = int(minA[0])
            peak_indices[i] = int(peak[0])
            ap1_widths[i]   = float(width[0])

    # ORIGINAL PRINT (kept)
    # print(f"True AP1_widths: {ap1_widths}. Shape: {ap1_widths.shape}")

    valid = (min_indices >= 0) & (peak_indices >= 0)
    batch_indices = torch.arange(batch_size, device=device)

    # Clamp indices to valid range
    min_cl = min_indices.clamp(0, time_steps - 1)
    peak_cl = peak_indices.clamp(0, time_steps - 1)

    min_voltage    = voltage[batch_indices, min_cl].squeeze(-1)  # (B,)
    peak_voltage   = voltage[batch_indices, peak_cl].squeeze(-1) # (B,)
    all_thresholds = (min_voltage + peak_voltage) / 2

    # Use EFEL thresholds where valid; others keep robust fallback
    thresholds[valid] = all_thresholds[valid]

    # NEW: for non-spiking trials, lift threshold above the true max to avoid full-window widths
    vmax = v_true_bt.max(dim=1).values  # (B,)
    thr_margin = torch.tensor(1.0, device=device)  # 1 mV margin (tune if needed)
    thresholds[~valid] = torch.maximum(
        thresholds[~valid],
        (vmax[~valid] + thr_margin).to(thresholds.dtype)
    )

    return thresholds, ap1_widths

def compute_predicted_and_true_ap1_width(
    stimulus: torch.Tensor,
    voltage_pred: torch.Tensor,
    voltage_true: torch.Tensor,
    time: torch.Tensor,
    k: float = 200.0,
    dt: float = 0.02, 
    ds: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute predicted and true AP1 widths for a batch (backward-stable; no DivBackward0).
    """
    true_voltage_threshold, true_ap1_widths = compute_threshold_voltage_and_ap1_width(stimulus, voltage_true, time, dt, ds)
    true_voltage_threshold = true_voltage_threshold.unsqueeze(-1)

    # ORIGINAL PRINT (kept)
    # print(f"Thresholds: {true_voltage_threshold}, shape: {true_voltage_threshold.shape}")

    # Stop grads into EFEL/true path
    true_voltage_threshold = true_voltage_threshold.detach()

    if voltage_pred.dim() == 3 and voltage_pred.shape[-1] == 1:
        voltage_pred = voltage_pred.squeeze(-1)

    n_batch, n_len = voltage_pred.shape
    device, dtype = voltage_pred.device, voltage_pred.dtype

    # Soft "above threshold" signal
    above_threshold_soft = torch.sigmoid(k * (voltage_pred - true_voltage_threshold))

    # Soft first up/down crossings via gated forward differences
    d_above = above_threshold_soft[..., 1:] - above_threshold_soft[..., :-1]  # (B, T-1)
    up_cross_signal = torch.relu(d_above)
    down_cross_signal = torch.relu(-d_above)

    up_count = torch.cumsum(up_cross_signal, dim=-1)
    up_gate = torch.sigmoid(-k * (up_count - 1.5))
    first_up_signal = up_cross_signal * up_gate

    down_count = torch.cumsum(down_cross_signal, dim=-1)
    down_gate = torch.sigmoid(-k * (down_count - 1.5))
    first_down_signal = down_cross_signal * down_gate

    # --- Division-free, masked interpolation ---
    v0 = voltage_pred[..., :-1]
    v1 = voltage_pred[..., 1:]
    dv = (v1 - v0).detach()                  # no grads through slope
    dv_abs = dv.abs()
    EPS_INTERP = 1e-4                        # raise if needed (e.g., 1e-3)
    # Only build 1/dv on steps that could matter:
    active = (first_up_signal + first_down_signal) > 0
    inv_dv = torch.zeros_like(dv)            # constant wrt grads
    inv_dv[active] = (dv.sign()[active] / torch.clamp(dv_abs[active], min=EPS_INTERP))

    # Numerator keeps grad wrt v0; denom is constant -> stable backward
    delta_t_raw = (true_voltage_threshold - v0) * inv_dv
    # keep within [0,1] and zero where inactive
    delta_t = torch.zeros_like(delta_t_raw)
    if delta_t.numel() > 0:
        delta_t[active] = torch.clamp(delta_t_raw[active], 0.0, 1.0)

    t_interp = torch.arange(n_len - 1, device=device, dtype=dtype).unsqueeze(0) + delta_t

    sum_up = torch.sum(first_up_signal, dim=-1, keepdim=True) + EPSILON
    up_weights = first_up_signal / sum_up

    sum_down = torch.sum(first_down_signal, dim=-1, keepdim=True) + EPSILON
    down_weights = first_down_signal / sum_down

    t_start = torch.sum(up_weights * t_interp, dim=-1)
    t_end = torch.sum(down_weights * t_interp, dim=-1)

    gate_up_sum = torch.sum(first_up_signal, dim=-1)
    gate_down_sum = torch.sum(first_down_signal, dim=-1)
    full_spike_gate = ((gate_up_sum > 0) & (gate_down_sum > 0)).to(dtype)

    width_in_steps = torch.relu(t_end - t_start) * full_spike_gate
    final_width = width_in_steps  # keep debugging honest

    return final_width * dt * ds, true_ap1_widths
