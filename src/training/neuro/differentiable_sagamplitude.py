import torch
import torch.nn.functional as F
import efel
from typing import Tuple


def get_start_and_end_times(stimulus: torch.Tensor, dt: float, ds: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the start and end times of the stimulus for each batch element.

    Args:
        stimulus: (B, T) tensor

    Returns:
        start_times: (B,) tensor of start indices
        end_times: (B,) tensor of end indices
    """
    device = stimulus.device
    # stimulus: (B, T)
    is_nonzero = stimulus != 0  # (B, T)
    B, T = stimulus.shape

    # Find first nonzero index in each row
    first_idx = is_nonzero.float().argmax(dim=1)  # (B,)

    # Find last nonzero index in each row
    # Flip along time, then find first nonzero (which is last in original)
    reversed_is_nonzero = torch.flip(is_nonzero, dims=[1])
    last_idx = T - 1 - reversed_is_nonzero.float().argmax(dim=1)

    # If all zero, set to 0 and T-1
    all_zero = (~is_nonzero.any(dim=1))
    first_idx = first_idx.masked_fill(all_zero, 0)
    last_idx = last_idx.masked_fill(all_zero, T - 1)

    return (first_idx * dt * ds).to(device), (last_idx * dt * ds).to(device)

def compute_true_sag_amplitude(input_batch, output_batch, time, dt=0.02, ds=3, max_time=515):
    """
    Compute the true sag amplitude for each batch element.

    Args:
        input_batch: (B, T) tensor
        output_batch: (B, T) tensor
        dt: float, default=0.02
        ds: int, default=3
        max_time: int, default=515

    Returns:
        sag_amplitude_arr: (B,) tensor of sag amplitudes
    """
    device = input_batch.device
    B = len(input_batch)
    
    # Get stimulus timing for all batch elements at once
    stim_start, stim_end = get_start_and_end_times(input_batch, dt, ds)
    
    # Convert all data to CPU and numpy at once (batch operation)
    input_np = input_batch.cpu().detach().numpy()
    output_np = output_batch.cpu().detach().numpy()
    time_np = time.cpu().detach().numpy()
    
    # Prepare all traces at once
    traces = []
    for i in range(B):
        trace = {
            'T': time_np,
            'V': output_np[i],
            'stim_start': [stim_start[i].cpu().item()],
            'stim_end': [stim_end[i].cpu().item()]
        }
        traces.append(trace)
    
    # Batch process all traces with efel
    spike_counts = efel.get_mean_feature_values(traces, ['Spikecount'])
    sag_amplitudes = efel.get_mean_feature_values(traces, ['sag_amplitude'])
    
    # Process results efficiently
    sag_amplitude_arr = torch.zeros(B, dtype=torch.float32, device=device)
    
    for i in range(B):
        if spike_counts[i]['Spikecount'] == 0:
            sag_amp = sag_amplitudes[i]['sag_amplitude']
            sag_amplitude_arr[i] = sag_amp if sag_amp is not None else 0.0
    
    return sag_amplitude_arr



def compute_differentiable_sag_amplitude(input_batch, output_batch, time, dt=0.02, ds=3, soft=True, temperature=50):
    """
    Compute the differentiable sag amplitude for each batch element.
    """
    device = input_batch.device
    B, T = output_batch.shape

    stim_start, stim_end = get_start_and_end_times(input_batch, dt, ds)

    stim_duration = stim_end - stim_start
    begin_time = stim_end - 0.1 * stim_duration
    end_time = stim_end

    # Expand time axis for broadcasting: (B, T)
    t_expanded = time.unsqueeze(0).expand(B, -1)

    # ---- steady-state mean near stim_end ----
    mask_ss = (t_expanded >= begin_time.unsqueeze(1)) & (t_expanded < end_time.unsqueeze(1))
    ss_count = mask_ss.float().sum(dim=1).clamp(min=1.0)  # (B,)
    steady_state_voltage = (output_batch * mask_ss.float()).sum(dim=1) / ss_count

    # ---- minimum during stimulus ----
    mask_stim = (t_expanded >= stim_start.unsqueeze(1)) & (t_expanded <= stim_end.unsqueeze(1))
    if soft:
        masked_voltage = output_batch + (~mask_stim) * 1e6
        weights = F.softmax(-masked_voltage * temperature, dim=1)  # softmin across T
        minimum_voltage = (weights * output_batch).sum(dim=1)
    else:
        masked_voltage = torch.where(mask_stim, output_batch, torch.full_like(output_batch, 1e6))
        minimum_voltage = masked_voltage.min(dim=1).values

    sag_amplitude = (steady_state_voltage - minimum_voltage)

    return sag_amplitude.to(device)