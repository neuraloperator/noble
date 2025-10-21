import torch, random, os, wandb, warnings
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from neuralop import LpLoss, H1Loss
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop.models import FNO

from training.data.create_dataloader import create_dataloader
from training.utils.training import log_training_setup
from training.utils.path_setup import setup_experiment_directories, build_model_string
from training.neuro.neuro_losses import compute_feature_loss
from training.neuro.neuron_model_utils import extract_scaled_e_features
from training.utils.embedding import build_input_batch_with_embeddings
from training.visualization.plotting import plot_samples
from training.neuro.extract_features import update_feature_errors


warnings.filterwarnings("ignore", category=RuntimeWarning, module="efel")

def compute_loss_metrics(pred: torch.Tensor, target: torch.Tensor, loss_fns: dict) -> dict:
    """
    Computes and returns all specified loss metrics.
    
    Args:
        pred (Tensor): Predicted output
        target (Tensor): Ground truth output
        loss_fns (dict): Dictionary of loss functions to compute
        
    Returns:
        dict: Dictionary of {loss_name: loss_value}
    """
    return {name: fn(pred, target).item() for name, fn in loss_fns.items()}

def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    Computes the L2 norm of the gradients of the model parameters.
    Args:
        model (torch.nn.Module): The model whose gradients to compute.
    Returns:
        float: The L2 norm of the gradients.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** 0.5
    return grad_norm

def relative_l2_loss(pred, target, eps=1e-8):
    """
    Computes the total squared relative L2 loss between pred and target.
    This is equivalent to ||pred - target||_2^2 / ||target||_2^2.
    """
    # Numerator: Sum of squared errors
    numerator = torch.sum((pred - target) ** 2)
    
    # Denominator: Sum of squared true values
    denominator = torch.sum(target**2) + eps
    
    return numerator / denominator

def l2_loss(pred, target, eps=1e-8):
    """
    Computes the total squared relative L2 loss between pred and target.
    This is equivalent to ||pred - target||_2^2.
    """
    return torch.sum((pred - target) ** 2)


def finetune_model(config: dict, wandb_run: wandb.sdk.wandb_run.Run) -> None:
    """
    This function trains NOBLE using the configuration parameters provided, and 
    logs the training progess using Weights and Biases.

    Args:
        config (dict): YAML configuration dictionary
        wandb_run (wandb.sdk.wandb_run.Run): Weights and Biases run object
    """
    # ============================================================================
    # INITIALIZATION AND SETUP
    # ============================================================================
    fno_config             = config['params']['network']
    embedding_config       = config['params']['embeddings']
    data_generation_config = config['params']['data_generation']
    train_config           = config['params']['training']
    finetune_config        = config['params']['finetune']
    optimizer_config       = config['params']['optimizer']
    log_config             = config['params']['logging']
    paths_config           = config['params']['paths']

    if finetune_config['feature_loss'] is False:
        raise ValueError("feature_loss must be set (not False) for finetuning. Please specify a feature loss in the config.")
    
    seed   = train_config['seed']
    device = train_config['device']

    random.seed(seed)
    torch.manual_seed(seed)

    # ============================================================================
    # DATA LOADER CREATION
    # ============================================================================
    if data_generation_config['window']:
        train_loader, test_loader = create_dataloader(
            config               = config,
            window_size_ms       = data_generation_config['signal_length'],
            train_max_start_ms   = 30.0,
            test_fixed_start_ms  = 15.0,
            dt                   = data_generation_config['dt'],
            ds_factor            = data_generation_config['ds_factor'],
            take_random_window   = True,
            window               = True)
    else:
        train_loader, test_loader = create_dataloader(
            config = config,
            window = False)
    
    # ============================================================================
    # FEATURE EMBEDDING SETUP
    # ============================================================================
    features_to_embed = embedding_config['e_features_to_embed']

    # Only extract features if we have features to embed
    if features_to_embed:
        normalised_features = extract_scaled_e_features(
            config=config,
            device=device,
            features_to_embed=features_to_embed,
            feature_range=(0.5, 3.5))

    # ============================================================================
    # MODEL ARCHITECTURE SETUP
    # ============================================================================
    in_channels = fno_config['in_channels']

    if embedding_config['sine_embeddings_freq'] is not None:
        in_channels      +=  2 * int(embedding_config['sine_embeddings_freq'])
    
    if embedding_config['hof_model_embeddings'] is not None and features_to_embed:
        in_channels += 2 * int(embedding_config['hof_model_embeddings']) * len(features_to_embed)

    norm = 'group_norm' if fno_config['group_norm'] else None

    fno_model = FNO(n_modes=(fno_config['n_modes'],),
                    in_channels=in_channels, 
                    out_channels=fno_config['out_channels'],
                    hidden_channels=fno_config['hidden_channels'],
                    n_layers=fno_config['n_layers'],
                    projection_channel_ratio=fno_config['projection_channel_ratio'],
                    norm=norm)

    fno_model = fno_model.to(device)

    wandb_run.summary["num_params"] = count_model_params(fno_model)

    # ============================================================================
    # PRETRAINED MODEL LOADING
    # ============================================================================
    pretrained_path = finetune_config['pretrained_model_path']

    if pretrained_path is False:
        raise ValueError("pretrained_model_path must be provided for finetuning. Please specify a pretrained model path in the config.")
    
    fno_model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))

    # ============================================================================
    # OPTIMIZER AND SCHEDULER SETUP
    # ============================================================================
    optimizable_parameters = fno_model.parameters()

    if optimizer_config['optimizer_name'].lower() == 'adamw':
        optimizer = AdamW(optimizable_parameters, lr=float(optimizer_config['lr']), weight_decay=float(optimizer_config['weight_decay']))
    
    elif optimizer_config['optimizer_name'].lower() == 'sgd':
        optimizer = torch.optim.SGD(optimizable_parameters, lr=float(optimizer_config['lr']), weight_decay=float(optimizer_config['weight_decay']))
                                    
    elif optimizer_config['optimizer_name'].lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(optimizable_parameters, lr=float(optimizer_config['lr']), history_size=optimizer_config['history_size'])
    
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_config['optimizer_name']}")

    scheduler = None

    if optimizer_config['scheduler'] == 'ReduceLROnPlateauTest' or optimizer_config['scheduler'] == 'ReduceLROnPlateauTrain':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, 
                                                               mode = 'min', 
                                                               factor = optimizer_config['scheduler_factor']/100, 
                                                               patience = optimizer_config['patience'])

    # ============================================================================
    # TRAINING CONFIGURATION AND LOSS SETUP
    # ============================================================================
    epochs       = train_config['epochs']
    train_loss   = train_config['train_loss']

    print_freq      = log_config['print_freq']
    plot_freq       = log_config['plot_freq']
    model_save_freq = log_config['model_save_freq'] 
    save_model      = log_config['save_model']

    base_losses = {
        'l1': LpLoss(d=1, p=1, reduction='sum'),
        'l2': LpLoss(d=1, p=2, reduction='sum'),
        'l4': LpLoss(d=1, p=4, reduction='sum'),
        'h1': H1Loss(d=1, reduction='sum')}

    train_loss_key = train_config['train_loss'].lower()
    loss_type_key  = train_config['train_loss_type'].lower()

    if train_loss_key not in base_losses:
        raise ValueError("Valid train_loss values in config are 'H1', 'L1', 'L2', or 'L4'.")

    if loss_type_key not in {'abs', 'rel'}:
        raise ValueError("Valid train_loss_type values in config are 'abs' or 'rel'.")

    train_loss = base_losses[train_loss_key]
    train_loss = train_loss if loss_type_key == 'rel' else train_loss.abs
    
    loss_metrics = {
        "H1_abs": base_losses['h1'].abs,
        "H1_rel": base_losses['h1'],
        "L1_abs": base_losses['l1'].abs,  
        "L1_rel": base_losses['l1'],         
        "L2_abs": base_losses['l2'].abs,
        "L2_rel": base_losses['l2'],
        "L4_abs": base_losses['l4'].abs,
        "L4_rel": base_losses['l4']}

    # ============================================================================
    # PATHS SETUP AND LOGGING
    # ============================================================================
    indices_for_plotting = random.sample(range(1, len(test_loader.dataset) + 1), 40)
    indices_for_plotting.sort()

    # Path for saving figures and models:
    model_base_path  = paths_config['model_path']  
    figure_base_path = paths_config['figure_path']  

    model_dir, figure_main_dir = setup_experiment_directories(model_base_path, figure_base_path)
    model_str = build_model_string(config, in_channels)

    log_training_setup(config=config, in_channels=in_channels, fno_model=fno_model, optimizer=optimizer, train_loss_key=train_loss_key, loss_type_key=loss_type_key, train_loader=train_loader, test_loader=test_loader)

    # ============================================================================
    # MAIN FINETUNING LOOP
    # ============================================================================
    lowest_test_l4_rel_loss = 1

    for epoch in range(epochs + 1):
        train_metrics  = {k: 0.0 for k in loss_metrics}
        total_loss_val_epoch   = 0.0
        total_feature_loss_epoch = 0.0
        feature_errors_train   = defaultdict(list)
        total_missed_firing_train, total_missed_non_firing_train = 0, 0

        fno_model.train()

        for idx, (input_batch, output_batch, identifier_batch) in enumerate(train_loader):
            input_batch  = input_batch.to(device).unsqueeze(1)
            output_batch = output_batch.to(device).unsqueeze(1)
            
            time = torch.linspace(0, data_generation_config['signal_length'], input_batch.shape[2]).to(device)

            input_batch_transformed = build_input_batch_with_embeddings(input_batch, identifier_batch, normalised_features, features_to_embed, embedding_config, device)

            optimizer.zero_grad()

            if optimizer_config['optimizer_name'].lower() == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    output_pred_batch = fno_model(input_batch_transformed)

                    feature_name = list(finetune_config['feature_loss'].keys())[0]
                    feature_loss = compute_feature_loss(stimulus=input_batch.squeeze(1),
                                                        output_pred=output_pred_batch.squeeze(1),
                                                        output_true=output_batch.squeeze(1),
                                                        feature_name=feature_name,
                                                        loss_fn=relative_l2_loss if loss_type_key == 'rel' else l2_loss,
                                                        time=time,
                                                        dt=data_generation_config['dt'],
                                                        ds=data_generation_config['ds_factor'])

                    feature_loss.backward()
                    return feature_loss

                loss_val = optimizer.step(closure)

                # Forward pass again for metrics (no gradients needed)
                with torch.no_grad():
                    output_pred_batch = fno_model(input_batch_transformed)
                    # Recompute feature loss for tracking
                    feature_name = list(finetune_config['feature_loss'].keys())[0]
                    feature_loss = compute_feature_loss(stimulus=input_batch.squeeze(1),
                                                        output_pred=output_pred_batch.squeeze(1),
                                                        output_true=output_batch.squeeze(1),
                                                        feature_name=feature_name,
                                                        loss_fn=relative_l2_loss if loss_type_key == 'rel' else l2_loss,
                                                        time=time,
                                                        dt=data_generation_config['dt'],
                                                        ds=data_generation_config['ds_factor'])

            else:
                # ---- (AdamW, SGD) ----
                output_pred_batch = fno_model(input_batch_transformed)

                data_loss  = train_loss(output_pred_batch, output_batch)

                feature_name = list(finetune_config['feature_loss'].keys())[0]
                feature_weight = finetune_config['feature_loss'][feature_name]
                
                feature_loss = compute_feature_loss(stimulus=input_batch.squeeze(1),
                                                    output_pred=output_pred_batch.squeeze(1),
                                                    output_true=output_batch.squeeze(1),
                                                    feature_name=feature_name,
                                                    loss_fn=relative_l2_loss if loss_type_key == 'rel' else l2_loss,
                                                    time=time,
                                                    dt=data_generation_config['dt'],
                                                    ds=data_generation_config['ds_factor'])

                loss_val = data_loss + feature_weight * feature_loss

                loss_val.backward()
                optimizer.step()

            total_loss_val_epoch += loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val
            total_feature_loss_epoch += feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss

            # ---- Data-based metrics ----
            batch_metrics = compute_loss_metrics(output_pred_batch, output_batch, loss_metrics)
            for k, v in batch_metrics.items():
                train_metrics[k] += v

            # ---- Feature-based metrics ----
            input_batch_np  = input_batch.squeeze(1).cpu().detach().numpy()
            output_batch_np = output_batch.squeeze(1).cpu().detach().numpy()
            output_pred_np  = output_pred_batch.squeeze(1).cpu().detach().numpy()

            batch_feature_errs, batch_missed_firing, batch_missed_non_firing = update_feature_errors(input_batch_np, output_batch_np, output_pred_np, data_generation_config)

            for k, v in batch_feature_errs.items():
                feature_errors_train[k].extend(v)

            total_missed_firing_train     += batch_missed_firing
            total_missed_non_firing_train += batch_missed_non_firing

            # ---- Plotting ----
            if epoch % plot_freq == 0:
                plot_samples(stimulus_batch=input_batch, 
                             true_response_batch=output_batch,
                             pred_response_batch=output_pred_batch,
                             figure_root=figure_main_dir, 
                             indices_for_plotting=indices_for_plotting,
                             mode='train',
                             epoch=epoch, 
                             FFT=True, 
                             idx_offset=idx,
                             data_config=data_generation_config)

        if scheduler and optimizer_config['scheduler'] == 'ReduceLROnPlateauTrain':
            scheduler.step(total_loss_val_epoch)

        # ========================================================================
        # EVALUATION PHASE
        # ========================================================================
        test_metrics = {k: 0.0 for k in loss_metrics}
        feature_errors_test = defaultdict(list)
        total_missed_firing_test, total_missed_non_firing_test = 0, 0

        with torch.no_grad():
            fno_model.eval()
            for idx, (input_batch, output_batch, identifier_batch) in enumerate(test_loader):
                input_batch  = input_batch.to(device).unsqueeze(1)
                output_batch = output_batch.to(device).unsqueeze(1)

                input_batch_transformed = build_input_batch_with_embeddings(input_batch, identifier_batch, normalised_features, features_to_embed, embedding_config, device)

                output_pred_batch = fno_model(input_batch_transformed)

                # ---- Data-based metrics ----
                batch_metrics = compute_loss_metrics(output_pred_batch, output_batch, loss_metrics)
                for k, v in batch_metrics.items():
                    test_metrics[k] += v

                # ---- Feature-based metrics ----
                input_batch_np  = input_batch.squeeze(1).cpu().detach().numpy()
                output_batch_np = output_batch.squeeze(1).cpu().detach().numpy()
                output_pred_np  = output_pred_batch.squeeze(1).cpu().detach().numpy()

                batch_feature_errs, batch_missed_firing, batch_missed_non_firing = update_feature_errors(input_batch_np, output_batch_np, output_pred_np, data_generation_config)

                for k, v in batch_feature_errs.items():
                    feature_errors_test[k].extend(v)
                
                total_missed_firing_test     += batch_missed_firing
                total_missed_non_firing_test += batch_missed_non_firing

                # ---- Plotting ----
                if epoch % plot_freq == 0:
                    plot_samples(stimulus_batch=input_batch, 
                                 true_response_batch=output_batch,
                                 pred_response_batch=output_pred_batch,
                                 figure_root=figure_main_dir, 
                                 indices_for_plotting=indices_for_plotting,
                                 mode='test', 
                                 epoch=epoch, 
                                 FFT=True, 
                                 idx_offset=idx,
                                 data_config=data_generation_config)

        # ========================================================================
        # COMPUTE DATA-BASED LOSS ACCUMULATION AND NORMALIZATION
        # ========================================================================
        num_train = len(train_loader.dataset)
        num_test  = len(test_loader.dataset)

        train_metrics = {k: v / num_train for k, v in train_metrics.items()}
        test_metrics  = {k: v / num_test for k, v in test_metrics.items()}

        if scheduler and optimizer_config['scheduler'] == 'ReduceLROnPlateauTest': # Only step if scheduler is not None
            scheduler.step(test_metrics['L4_rel'])

        # ========================================================================
        # MODEL SAVING
        # ========================================================================
        # Save models
        if save_model:
            if test_metrics['L4_rel'] <= lowest_test_l4_rel_loss:
                for fname in os.listdir(model_dir):
                    if "bestepoch" in fname:
                        os.remove(os.path.join(model_dir, fname))
                    
                model_path = model_dir + model_str + f'_bestepoch-{epoch}.pth'
                torch.save(fno_model.state_dict(), model_path)
                lowest_test_l4_rel_loss = test_metrics['L4_rel']
            
            if epoch % model_save_freq == 0:
                model_path = model_dir + model_str + f'_epoch-{epoch}.pth'
                torch.save(fno_model.state_dict(), model_path)

        # ========================================================================
        # LOGGING AND MONITORING with WandB
        # ========================================================================
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        wandb_run.log({"Optimizer/learning_rate": current_lr}, step=epoch)
        
        # Log gradient norm
        grad_norm = compute_grad_norm(fno_model)
        wandb_run.log({"Optimizer/grad_norm": grad_norm}, step=epoch)

        # Log data-based losses
        wandb_run.log({f"Losses/train_{k}_loss": v for k, v in train_metrics.items()}, step=epoch)
        wandb_run.log({f"Losses/test_{k}_loss": v for k, v in test_metrics.items()}, step=epoch)
        
        # Log feature-based losses
        feature_name, _ = next(iter(finetune_config['feature_loss'].items()))
        total_loss_metric = total_loss_val_epoch / num_train
        total_feature_loss_metric = total_feature_loss_epoch / num_train
        wandb_run.log({f"Losses/train_combined_{feature_name}_feature_L2_rel": total_loss_metric}, step=epoch)
        wandb_run.log({f"Losses/train_{feature_name}_feature_L2_rel": total_feature_loss_metric}, step=epoch)

        # Log electrophysiological errors
        avg_feature_errors_train = {f"Features/train_{feat} (%)": 100 * float(np.mean(errs)) for feat, errs in feature_errors_train.items()}
        wandb_run.log(avg_feature_errors_train, step=epoch)

        wandb_run.log({"Features/train_missed_firing": total_missed_firing_train, 
                       "Features/train_missed_non_firing": total_missed_non_firing_train}, step=epoch)

        avg_feature_errors_test = {f"Features/test_{feat} (%)": 100 * float(np.mean(errs)) for feat, errs in feature_errors_test.items()}
        wandb_run.log(avg_feature_errors_test, step=epoch)

        # Log the number of missed spiking/non-spiking predictions
        wandb_run.log({"Features/test_missed_firing": total_missed_firing_test, 
                       "Features/test_missed_non_firing": total_missed_non_firing_test}, step=epoch)

        # Print loss values
        if epoch % print_freq == 0:
            print(f"Epoch {epoch}: " + ", ".join([f"train_{k} = {train_metrics[k]:.7f}" for k in train_metrics]) + f", train_combined_{feature_name}_feature_L2_rel = {total_loss_metric:.7f}, train_{feature_name}_feature_L2_rel = {total_feature_loss_metric:.7f}")
            print(f"Epoch {epoch}: " + ", ".join([f"test_{k} = {test_metrics[k]:.7f}" for k in test_metrics]))
