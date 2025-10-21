import os
import numpy as np
import matplotlib.pyplot as plt
from training.utils.fft_utils import run_fft
import torch

def plot_response(
    stimulus: np.ndarray, 
    true_response: np.ndarray, 
    data_config: dict,
    predicted_response: np.ndarray = None, 
    path: str = None, 
    FFT: bool = False, 
    data_type: str = None
) -> None:
    """
    Plot input stimulus and neural response data for a single sample with optional FFT analysis.
    
    Args:
        stimulus (np.ndarray): Input stimulus signal
        true_response (np.ndarray): Ground truth neural response
        data_config: Configuration dictionary containing data generation parameters:
            - dt: Original time step in ms for generating the data pre-downsampling
            - ds_factor: Downsampling factor used in generating the training data
            - signal_length: Duration of stimulus and response signals in ms
        predicted_response (np.ndarray, optional): Predicted neural response. Defaults to None.
        path (str, optional): File path to save the plot. Defaults to None.
        FFT (bool, optional): Whether to include FFT analysis. Defaults to False.
        data_type (str, optional): Type of data being plotted. Defaults to None.
    """
    signal_duration = data_config['signal_length']
    dt = data_config['dt']
    ds_factor = data_config['ds_factor']

    time = np.linspace(0, signal_duration, len(stimulus))
    sampling_rate = 1000 / (dt * ds_factor)

    stimulus *= 1e-10

    if predicted_response is None:
        if FFT:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 8))

            # Time domain
            ax1.plot(time, true_response, color='cornflowerblue', label='True Response')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Response', color='cornflowerblue')
            ax1.tick_params(axis='y', labelcolor='cornflowerblue')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time, stimulus, color='orangered', label='Stimulus')
            ax1_twin.set_ylabel('Stimulus', color='orangered')
            ax1_twin.tick_params(axis='y', labelcolor='orangered')

            # FFT of true response
            fft_true, freq_true = run_fft(true_response, sampling_rate)
            ax2.plot(freq_true, np.abs(fft_true), color='cornflowerblue', alpha=0.7)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Log Magnitude')
            ax2.set_yscale('log')
            ax2.set_title('FFT of True Response')
            fig.tight_layout()

        else:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(time, true_response, color='cornflowerblue', label='True Response')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Response (V)', color='cornflowerblue')
            ax.tick_params(axis='y', labelcolor='cornflowerblue')
            ax_twin = ax.twinx()
            ax_twin.plot(time, stimulus, color='orangered', label='Stimulus')
            ax_twin.set_ylabel('Stimulus (A)', color='orangered')
            ax_twin.tick_params(axis='y', labelcolor='orangered')
            fig.tight_layout()

    else:
        if FFT:
            error_time = np.abs(predicted_response - true_response)

            fft_true, freq_true = run_fft(true_response, sampling_rate)
            fft_pred, freq_pred = run_fft(predicted_response, sampling_rate)

            error_fft = np.abs(np.abs(fft_pred) - np.abs(fft_true))

            # Create zoomed indices for frequencies with |f| < 100 Hz
            trunc_indices = np.where(np.abs(freq_true) < 100)[0]

            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

            # Column 1: True Response with Stimulus
            ax_true = axs[0, 0]
            ax_true.plot(time, true_response, color='cornflowerblue', label='True Response')
            ax_true.set_xlabel('Time (s)')
            ax_true.set_ylabel('Response (V)', color='cornflowerblue')
            ax_true.tick_params(axis='y', labelcolor='cornflowerblue')
            ax_true.set_title('True Response (Time Domain)')
            ax_true_twin = ax_true.twinx()
            ax_true_twin.plot(time, stimulus, color='orangered', label='Stimulus')
            ax_true_twin.set_ylabel('Stimulus (A)', color='orangered')
            ax_true_twin.tick_params(axis='y', labelcolor='orangered')

            # Column 2: Predicted Response with Stimulus
            ax_pred = axs[0, 1]
            ax_pred.plot(time, predicted_response, color='mediumseagreen', label='Predicted Response')
            ax_pred.set_xlabel('Time (s)')
            ax_pred.set_ylabel('Response (V)', color='mediumseagreen')
            ax_pred.tick_params(axis='y', labelcolor='mediumseagreen')
            ax_pred.set_title('Predicted Response (Time Domain)')
            ax_pred_twin = ax_pred.twinx()
            ax_pred_twin.plot(time, stimulus, color='orangered', label='Stimulus')
            ax_pred_twin.set_ylabel('Stimulus (A)', color='orangered')
            ax_pred_twin.tick_params(axis='y', labelcolor='orangered')

            # Column 3: Time Domain Error (Absolute Error between true and predicted)
            ax_error_time = axs[0, 2]
            ax_error_time.plot(time, error_time, color='red', label='Time Domain Error')
            ax_error_time.set_xlabel('Time (s)')
            ax_error_time.set_ylabel('Absolute Error (V)')
            ax_error_time.set_title('Time Domain Absolute Error')

            # ----- Middle Row: FFT Plots (Full Frequency Range) -----
            # Column 1: FFT of True Response
            ax_fft_true = axs[1, 0]
            ax_fft_true.plot(freq_true, np.abs(fft_true), color='cornflowerblue', alpha=0.7)
            ax_fft_true.set_xlabel('Frequency (Hz)')
            ax_fft_true.set_ylabel('Log Magnitude')
            ax_fft_true.set_yscale('log')
            ax_fft_true.set_title('FFT of True Response')

            # Column 2: FFT of Predicted Response
            ax_fft_pred = axs[1, 1]
            ax_fft_pred.plot(freq_pred, np.abs(fft_pred), color='mediumseagreen', alpha=0.7)
            ax_fft_pred.set_xlabel('Frequency (Hz)')
            ax_fft_pred.set_ylabel('Log Magnitude')
            ax_fft_pred.set_yscale('log')
            ax_fft_pred.set_title('FFT of Predicted Response')

            # Column 3: FFT Error (Absolute difference in FFT Magnitudes)
            ax_fft_error = axs[1, 2]
            ax_fft_error.plot(freq_true, error_fft, color='red', alpha=0.7)
            ax_fft_error.set_xlabel('Frequency (Hz)')
            ax_fft_error.set_ylabel('Absolute Error')
            ax_fft_error.set_yscale('log')
            ax_fft_error.set_title('FFT Absolute Error')

            # ----- Bottom Row: Zoomed FFT Plots (|Frequency| < 100 Hz) -----
            # Column 1: Zoomed FFT of True Response
            ax_zoom_true = axs[2, 0]
            ax_zoom_true.plot(freq_true[trunc_indices], np.abs(fft_true)[trunc_indices], color='cornflowerblue', alpha=0.7)
            ax_zoom_true.set_xlabel('Frequency (Hz)')
            ax_zoom_true.set_ylabel('Log Magnitude')
            ax_zoom_true.set_yscale('log')

            # Column 2: Zoomed FFT of Predicted Response
            ax_zoom_pred = axs[2, 1]
            ax_zoom_pred.plot(freq_pred[trunc_indices], np.abs(fft_pred)[trunc_indices], color='mediumseagreen', alpha=0.7)
            ax_zoom_pred.set_xlabel('Frequency (Hz)')
            ax_zoom_pred.set_ylabel('Log Magnitude')
            ax_zoom_pred.set_yscale('log')

            # Column 3: Zoomed FFT Error (Absolute Error in FFT, |f| < 100 Hz)
            ax_zoom_error = axs[2, 2]
            ax_zoom_error.plot(freq_true[trunc_indices], error_fft[trunc_indices], color='red', alpha=0.7)
            ax_zoom_error.set_xlabel('Frequency (Hz)')
            ax_zoom_error.set_ylabel('Absolute Error')
            ax_zoom_error.set_yscale('log')

            ax_pred.set_ylim(ax_true.get_ylim())
            ax_fft_pred.set_ylim(ax_fft_true.get_ylim())
            ax_zoom_pred.set_ylim(ax_zoom_true.get_ylim())

            fig.tight_layout()

        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
            ax1.plot(time, true_response, color='cornflowerblue', label='True Response')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Response', color='cornflowerblue')
            ax1.tick_params(axis='y', labelcolor='cornflowerblue')
            ax1.set_title('True Response')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time, stimulus, color='orangered', label='Stimulus')
            ax1_twin.set_ylabel('Stimulus', color='orangered')
            ax1_twin.tick_params(axis='y', labelcolor='orangered')

            ax2.plot(time, predicted_response, color='mediumseagreen', label='Predicted Response')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Response', color='mediumseagreen')
            ax2.tick_params(axis='y', labelcolor='mediumseagreen')
            ax2.set_title('Predicted Response')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(time, stimulus, color='orangered')
            ax2_twin.set_ylabel('Stimulus', color='orangered')
            ax2_twin.tick_params(axis='y', labelcolor='orangered')

            fig.tight_layout()

    if data_type is not None:
        fig.suptitle(f"Evaluation on {data_type.capitalize()}ing Data", fontsize=16, y=1.02)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')

    plt.close(fig)

def plot_samples(
    stimulus_batch: torch.Tensor,
    true_response_batch: torch.Tensor,
    data_config: dict,
    pred_response_batch: torch.Tensor = None,
    figure_root: str = None,
    indices_for_plotting: list = None,
    mode: str = 'train',
    idx_offset: int = 0,
    FFT: bool = False,
    epoch: int = None
) -> None:
    """
    Plot the input stimuli and responses for a batch of data.
    
    Args:
        stimulus_batch: Batch of stimulus signals
        true_response_batch: Batch of true response signals
        data_config: Configuration dictionary containing data generation parameters:
            - dt: Original time step in ms for generating the data pre-downsampling
            - ds_factor: Downsampling factor used in generating the training data
            - signal_length: Duration of stimulus and response signals in ms
        pred_response_batch: Batch of predicted response signals (optional)
        figure_root: Root directory for saving plots
        indices_for_plotting: List of indices of samples to plot (if None, plot all)
        mode: Plotting mode ('true', 'train', or 'test')
        idx_offset: Offset for batch indexing
        FFT: Whether to include FFT plots
        epoch: Current training epoch number
    """
    
    assert mode in ['true', 'train', 'test'], f"Invalid mode '{mode}'"

    if mode == 'true':
        plot_dir = os.path.join(figure_root, 'true')
    else:
        plot_dir = os.path.join(figure_root, mode, f'epoch_{epoch}')
    
    os.makedirs(plot_dir, exist_ok=True)

    batch_size = stimulus_batch.size(0)

    for i in range(batch_size):
        plot_idx = idx_offset * batch_size + i

        # If indices_for_plotting is not provided, plot all samples
        if indices_for_plotting is not None and plot_idx not in indices_for_plotting:
            continue

        stimulus = stimulus_batch[i].cpu().detach().numpy().reshape(-1)
        true_response = true_response_batch[i].cpu().detach().numpy().reshape(-1)
        predicted_response = None

        if pred_response_batch is not None:
            predicted_response = pred_response_batch[i].cpu().detach().numpy().reshape(-1)

        filepath = os.path.join(plot_dir, f'sample_{plot_idx}.png')

        plot_response(stimulus=stimulus, true_response=true_response, predicted_response=predicted_response, path=filepath, FFT=FFT, data_type=mode, data_config=data_config)