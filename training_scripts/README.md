# SLURM Training Scripts for NOBLE

These scripts are designed to launch reproducible large-scale $\texttt{NOBLE}$ training and finetuning jobs on GPU HPC clusters managed by `SLURM`, with advanced performance tracking using `Weights and Biases`.

This folder contains two example scripts for running $\texttt{NOBLE}$:

1. `submit_single_job.sh` — Submits a single $\texttt{NOBLE}$ training job  
2. `submit_single_finetune_job.sh` — Submits a single $\texttt{NOBLE}$ finetuning job on an electrophysiological feature  

The corresponding Python training scripts are located in `noble/src/`.

For details on all hyperparameters, their expected types, and default values, refer to:  
- `noble/src/training/train_noble.py`  
- `noble/src/training/train_noble_finetune.py`  

Below, we highlight the main configurable parameters used across both scripts.

---

## FNO Architecture Parameters

- `--n_modes` — Number of Fourier modes  
- `--hidden_channels` — Number of hidden channels  
- `--n_layers` — Number of FNO layers  
- `--projection_channel_ratio` — Ratio of projection channels to hidden channels  
- `--group_norm` — Whether to include group normalization  

**Example:**
```bash
--n_modes 256 --hidden_channels 24 --n_layers 12 --projection_channel_ratio 4 --group_norm False
```

## Optimizer Parameters

- `--optimizer_name` — Optimizer type (`adamw` or `lbfgs`)
- `--lr` — Learning rate
- `--weight_decay` — Weight decay (for `adamw`)
- `--history_size` — History size (for `lbfgs`)
- `--scheduler` — Learning rate scheduler (`ReduceLROnPlateauTrain`, `ReduceLROnPlateauTest`, or`null`)
- `--patience` — Patience for the ReduceLROnPlateau scheduler
- `--scheduler_factor` — Scheduler factor (e.g., 10 for a reduction factor of 0.1)

**Example:**
```bash
--optimizer_name adamw, --weight_decay 0 --scheduler ReduceLROnPlateauTrain --patience 8 --scheduler_factor 70
```


## Embedding parameters

- `--sine_embeddings_freq` — Number of frequencies for sinusoidal embedding of the stimulus current amplitude
- `--scale_sine_embeddings` — Scaling type for the sinusoidal embedding (`amp`, `freq`, or `none`)
- `--hof_model_embeddings` — Number of frequencies for sinusoidal embedding of neuron models
- `--e_features_to_embed` — Electrophysiological features of neuron models to embed

Note:
- `--sine_embeddings_freq` and `--hof_model_embeddings` accept an integer or `null` (for no embedding)
- `--e_features_to_embed` accepts a list of feature names

**Example:**
```bash
--sine_embeddings_freq 9 --scale_sine_embeddings freq --hof_model_embeddings 1 --e_features_to_embed: ["slope", "intercept"]
```

## Training parameters

- `--epochs` — Number of training epochs
- `--train_loss` — Loss function (L1, L2, L4, or H1)
- `--train_loss_type` — Loss type (rel for relative, abs for absolute)
- `--batch_size_train` — Batch size for training
- `--batch_size_test` — Batch size for testing
- `--custom_prefix` — Custom prefix for the WandB run name

**Example:**
```bash
--epochs 200 --train_loss L4 --train_loss_type rel --batch_size_train 64 --batch_size_test 64 --custom_prefix 'NewRun'
```

## Data parameters
- `--cell_name` — The name of the neuron family, and cell, separated by an underscore
- `--dt` — Timestep in ms used to generate the original dataset before downsampling
- `--ds_factor` — Downsampling factor used to generate the final training dataset
- `--signal_length` — Duration of stimulus and response signals in ms
- `--window`—- Whether to use windowing to augment training data

**Example:**
```bash
--cell_name PVALB_689331391 --dt 0.02 --ds_factor 3 --signal_length 515 --window True
```

## Logging parameters
- `--plot_freq` — Frequency (in epochs) to generate and save plots
- `--print_freq` — Frequency (in epochs) to print performance metrics
- `--save_model` — Whether to save model parameters
- `--model_save_freq` — Frequency (in epochs) to save model checkpoints

Note:
If `--save_model` is True, in addition to saving every `model_save_freq` epochs, the best-performing model so far is also saved automatically.

**Example:**
```bash
--plot_freq 50 --print_freq 1 --save_model True --model_save_freq 50
```

## Path parameters

- `--data_path` — Path to the `.pkl` file containing training and test data
- `--e_features_path` — Path to the `.csv` file containing the electrophysiological features for the embeddings
- `--model_path` — Path where $\texttt{NOBLE}$ model parameters will be saved
- `--figure_path` — Path where generated plots will be saved


## Finetuning parameters

To finetune a pre-trained $\texttt{NOBLE}$ model on an electrophysiological feature, the total loss is defined as:

$$
\mathcal{L}(\lambda) := \mathcal{L}_{\rm data} + \lambda \mathcal{L}_{\rm feature}.
$$

- `--pretrained_model_path` — Path to the pre-trained model parameters
- `--feature_loss` — JSON specifying the feature name and its weight $\lambda$

Note:
- Currently, the only available feature for finetuning is `sag_amplitude`.

**Example:**
```bash
--pretrained_model_path PATH --feature_loss '{"sag_amplitude": 25}'
```