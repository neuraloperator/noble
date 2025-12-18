import os
from datetime import datetime

def get_job_info() -> dict:
    """Extract SLURM job information from environment variables."""
    return {
        'slurm_job_id': os.getenv("SLURM_JOB_ID"),
        'run_index': os.getenv("SWEEP_RUN_INDEX"),
        'sweep_idx': os.getenv("SWEEP_ID")
    }

def generate_run_identifier() -> str:
    """
    Generate a unique identifier for this run. Uses SLURM job info if available, 
    otherwise falls back to timestamp.
    
    Format:
    - SLURM: "job-id-{slurm_job_id}-{run_index}" (run_index only if present)
    - No SLURM: "10:06:2025-14:30:25" (HH:MM:SS-DD:MM:YYYY)
    
    Returns:
        str: Run identifier
    """
    job_info = get_job_info()
    
    if job_info['slurm_job_id']:
        run_id = f"job-id-{job_info['slurm_job_id']}"
        
        # Add run index if doing serial runs on same allocation
        if job_info['run_index']:
            run_id += f"-{job_info['run_index']}"
            
        return run_id
    else:
        # No SLURM - use timestamp for local/interactive runs
        now = datetime.now()
        return now.strftime("%d-%m-%Y_%H:%M:%S")

def build_model_string(config: dict, in_channels: int) -> str:
    """
    Build descriptive name for saving model parameter files.
    
    Args:
        fno_config (dict): Configuration file
        in_channels (int): Number of input channels after embeddings
    
    Returns:
        str: Model identifier string
    """
    neuron     = config['params']['data_generation']['cell_name']
    
    fno_config = config['params']['network']
    model_str = (f'NOBLE_{neuron}_FNO_nmodes-{fno_config["n_modes"]}_in-{in_channels}_'
                f'out-{fno_config["out_channels"]}_nlayers-{fno_config["n_layers"]}_'
                f'projectionratio-{fno_config["projection_channel_ratio"]}_'
                f'hc-{fno_config["hidden_channels"]}')
    
    if fno_config.get('group_norm'):
        model_str += '_group-norm'
    
    embedding_config = config['params']['embeddings']
    
    if embedding_config.get('num_current_embeddings') is not None:
        current_freq = embedding_config["num_current_embeddings"]
        current_scale_type = embedding_config.get('type_current_embeddings')
        
        if current_scale_type == 'freq':
            model_str += f'_AmpEmbeddings-FreqScaledNeRF-nfreq-{current_freq}'

        elif current_scale_type == 'amp':
            model_str += f'_AmpEmbeddings-AmpScaledNeRF-nfreq-{current_freq}'

        else:
            model_str += f'_NeRF-embeddings-nfreq-{current_freq}'
    
    if embedding_config.get('num_hof_model_embeddings') is not None:
        hof_freq = int(embedding_config["num_hof_model_embeddings"])
        hof_scale_type = embedding_config.get('type_hof_model_embeddings')

        if hof_scale_type == 'freq':
            model_str += f'_HoFEmbeddings-FreqScaledNeRF-nfreq-{hof_freq}'

        elif hof_scale_type == 'amp':
            model_str += f'_HoFEmbeddings-AmpScaledNeRF-nfreq-{hof_freq}'

        else:
            model_str += f'_HoFEmbeddings-NeRF-nfreq-{hof_freq}'
    
    return model_str

def setup_experiment_directories(models_base_path: str, figures_base_path: str) -> tuple[str, str]:
    """
    Create directories for saving model parameter files and figures.
    
    Args:
        models_base_path (str): Base directory path for models
        figures_base_path (str): Base directory path for figures
    
    Returns:
        tuple: (model_dir, figure_main_dir)
    """
    run_id = generate_run_identifier()
    
    # Build directory paths by appending run ID to base paths
    model_dir = os.path.expanduser(f"{models_base_path}/{run_id}/")
    figure_main_dir = os.path.expanduser(f"{figures_base_path}/{run_id}/")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_main_dir, exist_ok=True)
    
    return model_dir, figure_main_dir

def build_wandb_run_name(config: dict, custom_prefix: str = None) -> str:
    """
    Build WandB run name from configuration file.
    
    Args:
        config (dict): Full configuration dictionary with 'params' containing
                      'network', 'optimizer', and 'training' sections
        custom_prefix (str, optional): Custom prefix to prepend to the run name
    
    Returns:
        str: Formatted run name for wandb
    """
    job_info      = get_job_info()
    network_cfg   = config['params']['network']
    embedding_cfg = config['params']['embeddings']
    optimizer_cfg = config['params']['optimizer']
    training_cfg  = config['params']['training']
    
    if custom_prefix:
        job_part = custom_prefix + "__"
    else:
        job_part = ""

    # Start with job info
    if job_info['slurm_job_id']:
        job_part += f"job{job_info['slurm_job_id']}"
        if job_info['run_index']:
            job_part += f"-{job_info['run_index']}"
    else:
        # Use timestamp if no SLURM
        now = datetime.now()
        job_part += now.strftime("local-%d-%m-%Y_%H:%M:%S")
    
    # Core network parameters
    save_name = f"{job_part}__nmodes-{network_cfg['n_modes']}_hc-{network_cfg['hidden_channels']}_nl-{network_cfg['n_layers']}"
    
    # Group norm
    if network_cfg.get('group_norm'):
        save_name += '_group_norm'
    
    # Scheduler configuration
    scheduler = optimizer_cfg.get('scheduler')
    lr = float(optimizer_cfg["lr"])
    
    if scheduler == 'ReduceLROnPlateauTest':
        factor = optimizer_cfg["scheduler_factor"] / 100
        patience = optimizer_cfg["patience"]
        save_name += f'_scheduler-RLonPTest_lr_{lr}_factor-{factor}_patience-{patience}'

    elif scheduler == 'ReduceLROnPlateauTrain':
        factor = optimizer_cfg["scheduler_factor"] / 100
        patience = optimizer_cfg["patience"]
        save_name += f'_scheduler-RLonPTrain_lr_{lr}_factor-{factor}_patience-{patience}'
        
    else:  # scheduler is None
        save_name += f'_scheduler-None_lr_{lr}'
    
    # Embedding configuration
    has_embeddings = False

    if embedding_cfg.get('num_current_embeddings') is not None:
        current_freq = embedding_cfg["num_current_embeddings"]
        current_scale_type = embedding_cfg.get('type_current_embeddings')
        
        if current_scale_type == 'freq':
            save_name += f'_AmpEmbed-FreqScaledNeRF-nfreq-{current_freq}'
        elif current_scale_type == 'amp':
            save_name += f'_AmpEmbed-AmpScaledNeRF-nfreq-{current_freq}'
        else:
            save_name += f'_Unscaled-NeRF-embeddings-nfreq-{current_freq}'
        
        has_embeddings = True
    
    if embedding_cfg.get('num_hof_model_embeddings') is not None:
        hof_freq = int(embedding_cfg["num_hof_model_embeddings"])
        hof_scale_type = embedding_cfg.get('type_hof_model_embeddings')

        if hof_scale_type == 'freq':
            save_name += f'_HoFEmbed-FreqScaledNeRF-nfreq-{hof_freq}'

        elif hof_scale_type == 'amp':
            save_name += f'_HoFEmbed-AmpScaledNeRF-nfreq-{hof_freq}'

        else:
            save_name += f'_HoFEmbed-UnscaledNeRF-nfreq-{hof_freq}'

        has_embeddings = True
    
    if not has_embeddings:
        save_name += '_no-embed'
    
    # Training loss configuration
    train_loss = training_cfg["train_loss"]
    train_loss_type = training_cfg["train_loss_type"]
    save_name += f'_{train_loss}{train_loss_type}-loss'
    
    return save_name