import argparse, wandb, yaml, shutil, os
from training.engine.noble_finetune import finetune_model
from training.utils.path_setup import build_wandb_run_name
import json

def get_args() -> argparse.Namespace:
    """
    This function is used to collect arguments passed from the command line

    Returns:
        argparse.Namespace: the arguments passed from the command line as a parser object
    """

    parser = argparse.ArgumentParser(description="Train NOBLE Model with WandB Logging")

    parser.add_argument("--cfg_path", type=str, required=True, help="Configuration file for training")

    ## FNO Parameters
    parser.add_argument("--n_modes", type=int, help="List of number of modes per dimension")
    parser.add_argument("--hidden_channels", type=int, help="Number of hidden channels")
    parser.add_argument("--n_layers", type=int, help="Number of layers")
    parser.add_argument("--projection_channel_ratio", type=int, help="Projection channel ratio")
    parser.add_argument("--group_norm", type=bool, help="Whether to apply group normalization or not") 
    parser.add_argument("--in_channels", type=int, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, help="Number of output channels")

    ## Optimizer Parameters
    parser.add_argument("--optimizer_name", type=str, help="Name of the optimizer. Possible options are 'adamw', 'lbfgs'")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--history_size", type=int, help="History size for LBFGS optimizer")
    parser.add_argument("--scheduler", type=str, help="Scheduler for optimizer. Possible options are 'ReduceLROnPlateauTrain', 'ReduceLROnPlateauTest', or 'null' for no scheduler")
    parser.add_argument("--patience", type=int, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_factor", type=int, help="Factor for ReduceLROnPlateau scheduler")

    ## Embedding Parameters
    parser.add_argument("--sine_embeddings_freq", type=int, help="Specifies whether to include sinusoidal embeddings. By default is None")
    parser.add_argument("--scale_sine_embeddings", type=str, help="By default is null. If 'amp', scale the sine embeddings by the amplitude of the signal. If 'frew', scale the frequency of the sine embeddings by the amplitude of the signal")
    parser.add_argument("--hof_model_embeddings", type=int, help="Whether to embed the Hall of Fame model. Either null, or number of frequencies to embed, scaled by hof_model firing curver shape parameters")
    parser.add_argument("--e_features_to_embed", type=str, help="Comma-separated list of electrophysiological features to embed (e.g., 'slope,intercept' or empty string for none)")
    
    ## Training Parameters
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--train_loss", type=str, help="Loss used for training. Possible options now are 'H1' 'L1' 'L2' and 'L4'")
    parser.add_argument("--train_loss_type", type=str, help="Loss type used for training. Possible options now are 'rel' and 'abs'")
    parser.add_argument("--batch_size_train", type=int, help="Batch size for training")
    parser.add_argument("--batch_size_test", type=int, help="Batch size for testing")
    parser.add_argument("--evaluate_features", type=str, help="Whether to evaluate eFEL feature based performance metrics or not (True/False)")
    parser.add_argument("--custom_prefix", type=str, default="", help="Custom prefix for the wandb run name")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Device for training")

    ## Data Parameters
    parser.add_argument("--cell_name", type=str, help="Name of the cell")
    parser.add_argument("--dt", type=float, help="Timestep (ms) used to generate the original dataset before downsampling")
    parser.add_argument("--ds_factor", type=int, help="Downsampling factor used to generate the final training dataset")
    parser.add_argument("--signal_length", type=float, help="Duration of stimulus and response signals in ms")
    parser.add_argument("--window", type=str, help="Whether to use windowing to augment training data (True/False)")
    
    ## Logging Parameters
    parser.add_argument("--print_freq", type=int, help="Interval of epochs for printing training performance")
    parser.add_argument("--plot_freq", type=int, help="Interval of epochs for plotting training performance")
    parser.add_argument("--save_model", type=bool, help="Whether model should be saved or not after training with state_dict")
    parser.add_argument("--model_save_freq", type=int, help="Interval of epochs for saving model")

    ## Path Parameters
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--e_features_path", type=str, help="Path to the electrophysiological features for embeddings")
    parser.add_argument("--model_path", type=str, help="Path to save the model")
    parser.add_argument("--figure_path", type=str, help="Path to save the figures")

    ## Finetuning Parameters
    parser.add_argument("--pretrained_model_path", type=str, help="Path to the pretrained model")
    parser.add_argument("--feature_loss", type=str, help="String of dictionary of feature names and their loss weights. Eg '{\'AP1_width\': 1.0}' or False to disable feature loss")
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    This function loads the configuration file from the path provided

    Args:
        config_path (str): String path to the configuration file

    Returns:
        dict: A YAML dictionary file containing the configuration parameters
    """

    with open(config_path) as file:
        return yaml.load(file, Loader=yaml.SafeLoader)

def update_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    This function updates the default configuration dictionary with command line arguments.

    Args:
        config (dict): YAML configuration dictionary
        args (argparse.Namespace): Parser object containing arguments passed from the command line

    Returns:
        dict: A new YAML configuration dictionary with the updated parameters
    """

    args_dict = vars(args)
    
    for key, value in args_dict.items():
        if key == "cfg_path" or value is None: 
            continue

        # Special handling for e_features_to_embed
        if key == "e_features_to_embed":
            if value.strip() == "":
                # Empty string means no features
                config["params"]["embeddings"][key] = []
            else:
                # Split comma-separated string into list
                config["params"]["embeddings"][key] = [feature.strip() for feature in value.split(",")]
            continue

        # Special handling for feature_loss (parse as JSON if string)
        if key == "feature_loss":
            if value is None or str(value).lower() in ("none", "false", ""):
                value = False
            elif isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception as e:
                    example = "--feature_loss '{\"AP1_width\": 1.0}'"
                    raise ValueError(
                        f"Could not parse --feature_loss as JSON. Please use valid JSON syntax. Error: {e}\nExample: {example}"
                    )

        # Special handling for scheduler (convert "null" to None)
        if key == "scheduler":
            if value is None or str(value).lower() in ("none", "null", ""):
                value = None

        for section in config.get("params", {}):
            if key in config["params"][section]:  
                config["params"][section][key] = value
                break 

    return config

def save_additional_scripts(scripts: list, run: wandb.sdk.wandb_run.Run) -> None:
    """
    This function saves additional scripts to the WandB code directory
    
    Args:
        scripts (list): A list of script names/paths to save (can include relative paths)
        run: WandB run object
    """
    print(f'Directory: {run.dir}')
    code_folder = os.path.join(run.dir, "code")
    
    os.makedirs(code_folder, exist_ok=True)
    
    for script in scripts:
        abs_script_path = os.path.abspath(script)
        print(f"Looking for script: {script} -> {abs_script_path}")
        
        if not os.path.exists(abs_script_path):
            print(f"Warning: Script {script} not found at {abs_script_path}")
            continue
        
        dest_filename = os.path.basename(script)
        dest_path = os.path.join(code_folder, dest_filename)
        
        print(f"Copying {abs_script_path} to {dest_path}")
        shutil.copy(abs_script_path, dest_path)
        wandb.save(dest_path, policy="now")

def main():
    args = get_args()

    config = load_config(args.cfg_path)
    config = update_config_with_args(config, args)

    wandb.login()

    wandb_run_name = build_wandb_run_name(config, custom_prefix=args.custom_prefix)

    run = wandb.init(project="NOBLE_PVALB_689331391",
                     name=wandb_run_name,
                     config=config,
                     save_code=False)

    scripts_to_save = ["training/finetune_noble.py", "training/engine/noble_finetune.py"]

    save_additional_scripts(scripts_to_save, run)

    code_folder = os.path.join(run.dir, "code")

    os.makedirs(code_folder, exist_ok=True)

    final_config_path = os.path.join(code_folder, "final_config.yaml")

    with open(final_config_path, "w") as f:
        yaml.dump(config, f)

    wandb.save(final_config_path, policy="now")

    finetune_model(config, run)
    
    run.finish()

if __name__ == "__main__":
    main()
