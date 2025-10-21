import matplotlib.pyplot as plt
import numpy as np
import os


def plot_ensemble_prediction(predicted_output, num_samples, dt_downsampled, save_title):
    fig, ax = plt.subplots(figsize=(8, 5))
    time = np.arange(0, predicted_output.shape[1]) * dt_downsampled
    for sample in predicted_output:
        plt.plot(time, sample * 1000, label="Predicted output", c='salmon', linewidth=0.3)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.tight_layout()

    results_dir = "results"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    save_path = os.path.join(results_dir, f"{save_title}.png")
    plt.savefig(save_path)
    plt.show()

def plot_fi_curve(fi_curves, amplitudes):
    plt.figure(figsize=(6, 4))
    for model_fi in fi_curves:
        plt.plot(amplitudes/10, model_fi, c="k", alpha=0.5)
    
    plt.xlabel("Amplitude (nA)")
    plt.ylabel(f'Raw Spikecount')
    plt.title(f"Current Injection Amplitude vs Spikecount")
    plt.grid(True)

    results_dir = "results"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    save_path = os.path.join(results_dir, f"fi_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()