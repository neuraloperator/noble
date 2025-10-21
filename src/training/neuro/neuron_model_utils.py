import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def extract_scaled_e_features(config: dict, device: str, features_to_embed: list, feature_range: tuple = (0.5, 3.5)) -> pd.DataFrame: 
    """    
    This function reads electrophysiological features from a CSV file where each row represents
    a neuron model and each column represents a feature. It filters the data based on the neuron
    that NOBLE is being trained on, and scales the specified parsed features to a given range 
    using MinMaxScaler. Finally, it indexes the dataframe using a neuron models' Cell_Type, Cell_ID,
    Seed, and HoF, to be compatable with the identifier array in the dataloaders
    
    Args:
        config (dict): Configuration dictionary containing:
            - params.paths.e_features_path: Path to the CSV file containing electrophysiological features
            - params.data_generation.cell_name: Neuron identifier in format 'cell_type' or 'cell_type_cell_id'
        device (str): Device to load tensors on ('cpu' or 'cuda')
        features_to_embed (list): List of feature column names to extract and scale
        feature_range (tuple, optional): Range for MinMaxScaler (min, max). Defaults to (0.5, 3.5).
    
    Returns:
        pd.DataFrame: Indexed DataFrame containing scaled electrophysiological features with
                     multi-index (cell_type, cell_id, seed, hof) and scaled feature columns.
    """
    path = os.path.expanduser(config['params']['paths']['e_features_path'])
    complete_df = pd.read_csv(path)

    neuron_identifier = config['params']['data_generation']['cell_name']
    
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

    indexed_scaled_neuron_df = scaled_neuron_df[columns_to_extract].set_index(['cell_type', 'cell_id', 'seed', 'hof'])

    print("Neuron Feature Dataframe:")
    print(indexed_scaled_neuron_df, "\n")

    return indexed_scaled_neuron_df

def construct_e_feature_dict_batched_embedding(indexed_df: pd.DataFrame, identifiers: np.ndarray) -> dict[str, torch.Tensor]:
    """
    Construct a dictionary of feature tensors for batched embedding from DataFrame of electrophysiological features.
    
    Args:
        indexed_df (pd.DataFrame): Indexed DataFrame containing scaled electrophysiological features
        identifiers (np.ndarray): Array of neuron identifiers in format 'cell_type-cell_id-seed-hof'
    
    Returns:
        dict[str, torch.Tensor]: Dictionary mapping feature names to batched feature tensors
    """
    index_tuples = []
    for identifier in identifiers:
        if not isinstance(identifier, str):
            identifier_str = str(identifier)
        else:
            identifier_str = identifier
        # If identifier contains '_', replace with '-' to standardize splitting
        if '_' in identifier_str:
            identifier_str = identifier_str.replace('_', '-')
        
        parts = identifier_str.split('-')
        
        if len(parts) < 4:
            raise ValueError(f"Identifier '{identifier_str}' does not have the expected 4 parts separated by '-' or '_'")
        index_tuples.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
    
    non_feature_columns = ["cell_type", "cell_id", "seed", "hof"]
    feature_columns = [col for col in indexed_df.columns if col not in non_feature_columns]
    
    # Get the batch data
    batch_data = indexed_df.loc[index_tuples, feature_columns]
    
    # Convert each feature to a tensor
    feature_tensors = {}
    for feature in feature_columns:
        feature_tensors[feature] = torch.tensor(batch_data[feature].values, dtype=torch.float32)
    
    return feature_tensors
