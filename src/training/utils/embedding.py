import torch
from neuralop.layers.embeddings import SinusoidalEmbedding
from training.neuro.neuron_model_utils import construct_e_feature_dict_batched_embedding


def Construct_Embedded_Nerf_Batch(input_batch: torch.Tensor, 
                                  num_frequencies: int, 
                                  device: str, 
                                  scale_amp: bool = False, 
                                  scale_freq: bool = False, 
                                  embed_amp: bool = False, 
                                  embed_model: bool = False, 
                                  feature_tensor: torch.Tensor = None) -> torch.Tensor:
    """
    Construct embedded NeRF batch with optional amplitude and neuron model embeddings.
    
    Args:
        input_batch (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length)
        num_frequencies (int): Number of frequency components for NeRF embedding
        device (str): Device to place tensors on ('cpu' or 'cuda')
        scale_amp (bool): Whether to scale the amplitudes of the embeddings
        scale_freq (bool): Whether to scale the frequency of the embeddings
        embed_amp (bool): Whether to embed amplitude information
        embed_model (bool): Whether to embed neuron model information
        feature_tensor (torch.Tensor, optional): Tensor containing electrophysiological features for embedding
        
    Returns:
        torch.Tensor: Embedded tensor of shape (batch_size, embedded_channels, sequence_length)
    """

    batch_size, _, n_in = input_batch.shape 
    if embed_amp and embed_model:
        raise ValueError("Cannot embed both the amplitude and neuron model simultaneously")
    
    if embed_model and feature_tensor is None:
        raise ValueError("Missing tensor containing electrophysiological features for neuron model embedding. Ensure embed_model=True and provide feature_tensor parameter.")
    
    if embed_amp:
        input_perm = input_batch.permute(0, 2, 1)  # Only permute when needed for amplitude extraction
        scaling_tensor = torch.stack([row[row != 0][0] for row in input_perm.squeeze(-1)]).to(device)  # → (B,)
    
    if embed_model:
        scaling_tensor = feature_tensor.to(device)

    if scale_amp and scale_freq:
        raise ValueError("Cannot do both amplitude and frequency scaling simultaneously.")
    
    embedder = SinusoidalEmbedding(in_channels=1, 
                                   num_frequencies=num_frequencies,
                                   embedding_type='nerf').to(device)
                                
    # Create time coordinates from 0 to 2π
    time = torch.linspace(0, 2*torch.pi, n_in, device=device, dtype=torch.float32).unsqueeze(-1)  # (L, 1)
    time = time.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, 1) -> Required shape for SinusoidalEmbedding 
    
    # Apply scaling based on type
    if scale_freq:
        scaling_expanded = scaling_tensor.view(batch_size, 1, 1) # (B, 1, 1)
        # Scale time coordinates to modulate frequency
        time_scaled = time * scaling_expanded
        embed = embedder(time_scaled)
        
    elif scale_amp:
        embed = embedder(time)
        scaling_expanded = scaling_tensor.view(batch_size, 1, 1) # (B, 1, 1)
        embed = embed * scaling_expanded
        
    else:
        # No scaling
        embed = embedder(time)

    # Embedder outputs shape (B, L, C') -> Permute to (B, C', L)
    return embed.permute(0, 2, 1)

def build_input_batch_with_embeddings(input_batch: torch.Tensor, identifier_batch: torch.Tensor, normalised_features, features_to_embed, embedding_config: dict,  device: str) -> torch.Tensor:
    """
    This function augments the input batch with NeRF-style embeddings, including:
      - The amplitude of the input signal
      - Electrophysiological features of the hall of fame model
    with optional amplitude or frequency scaling.

    Args:
        input_batch (torch.Tensor): Input tensor of shape (batch_size, C, T), where C is the number of channels and T is the signal length.
        identifier_batch (torch.Tensor): Batch of identifiers for each sample, used to look up features for embedding.
        normalised_features: DataFrame or similar structure containing normalized features for embedding.
        features_to_embed (list): List of feature names to embed from the normalised_features.
        embedding_config (dict): Configuration dictionary specifying which embeddings to use and their parameters.
        device (str): Device to place the output tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The transformed input batch with all specified embeddings concatenated along the channel dimension.
    """
    input_batch_transformed = input_batch

    if embedding_config['sine_embeddings_freq'] is not None:

        if embedding_config['scale_sine_embeddings'] == 'amp':
            sine_embed = Construct_Embedded_Nerf_Batch(input_batch, num_frequencies=int(embedding_config['sine_embeddings_freq']), embed_amp=True, scale_amp=True, device=device)
        
        elif embedding_config['scale_sine_embeddings'] == 'freq':
            sine_embed = Construct_Embedded_Nerf_Batch(input_batch, num_frequencies=int(embedding_config['sine_embeddings_freq']), embed_amp=True, scale_freq=True, device=device)

        else:
            sine_embed = Construct_Embedded_Nerf_Batch(input_batch, num_frequencies=int(embedding_config['sine_embeddings_freq']), device=device)

        input_batch_transformed = torch.cat((input_batch_transformed, sine_embed), dim=1)
    
    if embedding_config['hof_model_embeddings'] is not None and features_to_embed:
        batched_embedding_feats = construct_e_feature_dict_batched_embedding(normalised_features, identifier_batch)

        feature_embeds = []

        for _, feature_tensor in batched_embedding_feats.items():

            embed = Construct_Embedded_Nerf_Batch(input_batch=input_batch,
                                                    num_frequencies=int(embedding_config['hof_model_embeddings']),
                                                    device=device,
                                                    embed_model=True,
                                                    scale_freq=True,
                                                    feature_tensor=feature_tensor)

            feature_embeds.append(embed)

        input_batch_transformed = torch.cat([input_batch_transformed] + feature_embeds, dim=1)

    return input_batch_transformed