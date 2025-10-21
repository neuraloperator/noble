import sys
from pathlib import Path
import torch
import numpy as np
from typing import Optional
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Add src to path for training imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from training.utils.embedding import Construct_Embedded_Nerf_Batch, construct_e_feature_dict_batched_embedding

def extract_scaled_e_features(neuron_identifier, path_to_features, features_to_embed: list, feature_range: tuple = (0.5, 3.5)) -> pd.DataFrame: 
    """    
    This function reads electrophysiological features from a CSV file where each row represents
    a neuron model and each column represents a feature. It filters the data based on the neuron
    that NOBLE is being trained on, and scales the specified parsed features to a given range 
    using MinMaxScaler. Finally, it indexes the dataframe using a neuron models' Cell_Type, Cell_ID,
    Seed, and HoF, to be compatable with the identifier array in the dataloaders
    
    Args:
        config (dict): Configuration dictionary containing:
            - params.paths.e_features_path: Path to the CSV file containing electrophysiological features
            - params.neuron.name: Neuron identifier in format 'cell_type' or 'cell_type_cell_id'
        device (str): Device to load tensors on ('cpu' or 'cuda')
        features_to_embed (list): List of feature column names to extract and scale
        feature_range (tuple, optional): Range for MinMaxScaler (min, max). Defaults to (0.5, 3.5).
    
    Returns:
        pd.DataFrame: Indexed DataFrame containing scaled electrophysiological features with
                     multi-index (cell_type, cell_id, seed, hof) and scaled feature columns.
    """
    path = os.path.expanduser(path_to_features)
    complete_df = pd.read_csv(path)
    
    if '_' in neuron_identifier:
        parts = neuron_identifier.split('_')
    else:
        parts = neuron_identifier.split('-')
    
    cell_type = parts[0]

    # Filter by cell type
    neuron_df = complete_df[complete_df['cell_type'] == cell_type].copy()

    # If cell_id is provided, filter further
    if len(parts) > 1:
        cell_id = int(parts[1])
        neuron_df['cell_id'] = neuron_df['cell_id'].astype(int)
        neuron_df = neuron_df[neuron_df['cell_id'] == cell_id].copy()

    if neuron_df.empty:
        raise ValueError(f"No data found for neuron identifier: {neuron_identifier}")
    
    columns_to_extract = ["cell_type", "cell_id", "seed", "hof"]
    columns_to_extract.extend(features_to_embed)
    
    scaled_neuron_df = neuron_df.copy()

    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_neuron_df[features_to_embed] = scaler.fit_transform(scaled_neuron_df[features_to_embed])

    final_df = scaled_neuron_df[columns_to_extract].copy()
    final_df["hof_model"] = ((final_df["seed"] - 1) * 20) + final_df["hof"]

    cols = final_df.columns.tolist()
    hof_idx = cols.index("hof")

    cols.insert(hof_idx + 1, cols.pop(cols.index("hof_model")))
    final_df = final_df[cols]

    return final_df

def build_input(amplitude, n, device):
    dt = 0.06
    L = int(515 / dt)
    pre_stim_steps  = int(np.round(15 / dt))
    stim_steps      = int(np.round(400 / dt))

    input_tensor_1d = torch.zeros(L, dtype=torch.float32)
    input_tensor_1d[pre_stim_steps:pre_stim_steps+stim_steps] = amplitude
    input_batch = input_tensor_1d.repeat(n, 1)
    input_batch = input_batch.to(device).unsqueeze(1)
    return input_batch

def build_input_with_embeddings(input_batch, embedding_config, features_to_embed, normalised_features, device, sampled_embeddings):
    """Helper: builds input with sine, amplitude, and model embeddings."""
    input_batch_transformed = input_batch

    # ---- Sine embeddings ----
    if embedding_config['sine_embeddings_freq'] is not None:
        if embedding_config['scale_sine_embeddings'] == 'amp':
            sine_embed = Construct_Embedded_Nerf_Batch(
                input_batch, num_frequencies=int(embedding_config['sine_embeddings_freq']),
                embed_amp=True, scale_amp=True, device=device
            )
        elif embedding_config['scale_sine_embeddings'] == 'freq':
            sine_embed = Construct_Embedded_Nerf_Batch(
                input_batch, num_frequencies=int(embedding_config['sine_embeddings_freq']),
                embed_amp=True, scale_freq=True, device=device
            )
        else:
            sine_embed = Construct_Embedded_Nerf_Batch(
                input_batch, num_frequencies=int(embedding_config['sine_embeddings_freq']),
                device=device
            )
        input_batch_transformed = torch.cat((input_batch_transformed, sine_embed), dim=1)

    # ---- HoF model embeddings ----
    if embedding_config['hof_model_embeddings'] is not None and features_to_embed:
        batched_embedding_feats = sampled_embeddings
        feature_embeds = []
        for _, feature_tensor in batched_embedding_feats.items():
            embed = Construct_Embedded_Nerf_Batch(
                input_batch=input_batch,
                num_frequencies=int(embedding_config['hof_model_embeddings']),
                device=device,
                embed_model=True,
                scale_freq=True,
                feature_tensor=feature_tensor)

            feature_embeds.append(embed)
        input_batch_transformed = torch.cat([input_batch_transformed] + feature_embeds, dim=1)

    return input_batch_transformed

    