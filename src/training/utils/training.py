from neuralop.utils import count_model_params
import torch


def log_training_setup(
    config: dict,
    in_channels: int,
    fno_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss_key: str,
    loss_type_key: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader) -> None:
    """
    Log the training setup configuration including model, data, embeddings, and optimizer details.
    
    Args:
        config (dict): Configuration dictionary containing all training parameters
        in_channels (int): Number of input channels for the model
        fno_model (torch.nn.Module): The FNO model instance
        optimizer (torch.optim.Optimizer): The optimizer instance
        train_loss_key (str): Key identifying the training loss function
        loss_type_key (str): Key identifying the loss type
        train_loader (torch.utils.data.DataLoader): Training data loader
        test_loader (torch.utils.data.DataLoader): Testing data loader
    """
    print('\n========== NEW TRAINING SESSION ==========\n')
    if config['params']['finetune']['pretrained_model_path'] is not False:
        feature_name = list(config['params']['finetune']['feature_loss'].keys())[0]
        feature_weight = config['params']['finetune']['feature_loss'][feature_name]
        print(f"Finetuning NOBLE on {config['params']['data_generation']['cell_name']} with {feature_name} feature loss with weight {feature_weight}\n")

    else:
        print(f"Training NOBLE on {config['params']['data_generation']['cell_name']}\n")
    
    print("### DATA ###")
    print(f"Data path: {config['params']['paths']['data_path']}")
    print(f"E-features path: {config['params']['paths']['e_features_path']}")
    print(f"Device for training: {config['params']['training']['device']}")
    print(f"Length of training dataset: {len(train_loader.dataset)}")
    print(f"Length of testing dataset: {len(test_loader.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}\n")


    print("\n### MODEL ###")
    fno_config = config['params']['network']
    if config['params']['finetune']['pretrained_model_path'] is not False:
        print(f"Will finetune the checkpoint located at {config['params']['finetune']['pretrained_model_path']}")
    print(f"Model architecture: {fno_model}")
    print(f"Number of modes: {fno_config['n_modes']}")
    print(f"Input channels: {in_channels}")
    print(f"Output channels: {fno_config['out_channels']}")
    print(f"Hidden channels: {fno_config['hidden_channels']}")
    print(f"Number of layers: {fno_config['n_layers']}")
    print(f"Projection channel ratio: {fno_config['projection_channel_ratio']}")
    print(f"Group norm: {fno_config['group_norm']}\n")
    print(f"Total trainable parameters: {count_model_params(fno_model)}")

    print("\n### EMBEDDINGS ###")
    ## Encoding the stimulus current amplitude
    embedding_config = config['params']['embeddings']
    if embedding_config['num_current_embeddings'] is not None:
        print(f"Encoding the stimulus current amplitude with {embedding_config['num_current_embeddings']} sinusoidal embedding frequencies")
        
        if embedding_config['type_current_embeddings'] == 'amp':
            print(f"Scaling the amplitude of the sinusoidal embeddings by the amplitude of the stimulus current")

        elif embedding_config['type_current_embeddings'] == 'freq':
            print(f"Scaling the frequency of the sinusoidal embeddings by the amplitude of the stimulus current")
    
    ## Encoding the Hall of Fame neuron model
    if embedding_config['num_hof_model_embeddings'] is not None and embedding_config['e_features_to_embed']:
        e_feats_to_embed = embedding_config['e_features_to_embed']
        if len(e_feats_to_embed) == 1:
            e_feat_str = e_feats_to_embed[0]
        elif len(e_feats_to_embed) == 2:
            e_feat_str = f"{e_feats_to_embed[0]} and {e_feats_to_embed[1]}"
        else:
            e_feat_str = f"{', '.join(e_feats_to_embed[:-1])}, and {e_feats_to_embed[-1]}"
        
        hof_type = embedding_config.get('type_hof_model_embeddings')
        print(f"Encoding the {e_feat_str} for each HoF neuron model with {embedding_config['num_hof_model_embeddings']} sinusoidal embedding frequencies")
        
        if hof_type == 'amp':
            print(f"Scaling the amplitude of the sinusoidal embeddings by the amplitude of {e_feat_str}")
        
        elif hof_type == 'freq':
            print(f"Scaling the frequency of the sinusoidal embeddings by {e_feat_str}")
            
    print("\n### OPTIMIZER ###")
    print(optimizer)
    
    optimizer_config = config['params']['optimizer']
    scheduler_type = optimizer_config['scheduler']
    
    if scheduler_type == 'ReduceLROnPlateauTest' or scheduler_type == 'ReduceLROnPlateauTrain':
        print(f"\nScheduler: {scheduler_type}")
        print(f"  - Initial LR: {optimizer_config['lr']}")
        print(f"  - Factor: {optimizer_config['scheduler_factor'] / 100}")
        print(f"  - Patience: {optimizer_config['patience']}")
    if scheduler_type is None:
        print(f"\nScheduler: None")
        print(f"  - LR: {optimizer_config['lr']}")

    print("\n### LOSSES ###")
    train_config = config['params']['training']

    print(f"Training data loss: {train_loss_key.upper()}_{loss_type_key}")

    if config['params']['finetune']['pretrained_model_path'] is not False:
        ## Finetuning pre-trained model on L2 loss on feature
        feature_name, feature_loss_weight = next(iter(config['params']['finetune']['feature_loss'].items()))
        print(f"Additional {feature_name} feature loss is included with weight {feature_loss_weight}")
 
    print("\n### TRAINING CONFIG ###")
    print(f"Epochs: {train_config['epochs']}")
    print(f"Evaluate features: {train_config['evaluate_features']}")
    print(f"Batch size (train): {train_config['batch_size_train']}")
    print(f"Batch size (test): {train_config['batch_size_test']}")
    print(f"Random seed: {train_config['seed']}")
    if config['params']['data_generation']['window']:
        print('Augmenting training data with random windows')
    else:
        print('Not augmenting training data with random windows')
    

    print("\n### LOGGING CONFIG ###")
    log_config = config['params']['logging']
    print(f"Print frequency: {log_config['print_freq']}")
    print(f"Plot frequency: {log_config['plot_freq']}")
    print(f"Model save frequency: {log_config['model_save_freq']}")
    print(f"Save model: {log_config['save_model']}")

    print("\n### SAVE DIRECTORIES ###")
    print(f"Model path: {config['params']['paths']['model_path']}")
    print(f"Figure path: {config['params']['paths']['figure_path']}")
    
    print('\n==========================================\n')
