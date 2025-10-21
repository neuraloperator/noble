from utils.input_builder import build_input, build_input_with_embeddings
import torch
import numpy as np
import efel
import gc
import pandas as pd


def generate_frequency_current_curve(amplitudes: np.ndarray, 
                                     sampled_models: dict, 
                                     noble_model: torch.nn.Module,
                                     normalised_features: pd.DataFrame, 
                                     features_to_embed: list, 
                                     embedding_config: dict, 
                                     device: str) -> np.ndarray:
    
    num_models = len(sampled_models[features_to_embed[0]])

    fi_curves = np.zeros((num_models, len(amplitudes)))

    last_printed = 0

    print("processed amplitudes 0%")  # initial message

    for amp_idx, amplitude in enumerate(amplitudes):
        percent_complete = int((amp_idx + 1) / len(amplitudes) * 100)

        # Only print when we cross a new 10% bucket
        if percent_complete // 10 > last_printed // 10:
            print(f"processed amplitudes {percent_complete}%")
            last_printed = percent_complete
            
        input_batch = build_input(amplitude, num_models, device)
        input_batch_transformed = build_input_with_embeddings(input_batch=input_batch, 
                                                              embedding_config=embedding_config, 
                                                              features_to_embed=features_to_embed, 
                                                              normalised_features=normalised_features, 
                                                              device=device,
                                                              sampled_embeddings=sampled_models)
        with torch.no_grad():
            output_batch = noble_model(input_batch_transformed)
            
        traces = []
        time = np.linspace(0, 515, input_batch.shape[2])

        for model_idx, (input_sample, output_sample) in enumerate(zip(input_batch, output_batch)):
            output = output_sample.squeeze(0).cpu().detach().numpy()

            trace = {
                'T': time,
                'V': output * 1000,
                'stim_start': [15],
                'stim_end': [415]
            }
            traces.append(trace)

        feature_values_efel = efel.get_mean_feature_values(traces, ["Spikecount"])

        feature_values = np.array([result["Spikecount"] if result["Spikecount"] is not None else 0 for result in feature_values_efel])
        
        fi_curves[:, amp_idx] = feature_values

        del input_batch_transformed, output_batch, traces, feature_values_efel, feature_values
        torch.cuda.empty_cache()
        gc.collect()

    return fi_curves