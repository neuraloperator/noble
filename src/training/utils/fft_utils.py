import numpy as np
import torch

def run_fft(signal: np.ndarray, sampling_freq: float, norm: str = 'forward') -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform of a signal and return the shifted FFT and frequency array.
    
    Args:
        signal (np.ndarray): Input signal array
        sampling_freq (float): Sampling frequency of the signal
        norm (str): Normalization mode for FFT computation (Defaults to 'forward')
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the shifted FFT array and frequency array
    """
    signal_fft       = np.fft.fft(signal, axis=0, norm=norm)
    signal_fft_shift = np.fft.fftshift(signal_fft)

    signal_freq = np.fft.fftfreq(signal_fft.shape[0], d=1/sampling_freq)
    signal_freq = np.fft.fftshift(signal_freq)

    return signal_fft_shift, signal_freq