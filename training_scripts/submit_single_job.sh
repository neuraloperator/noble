#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J "NOBLE"
#SBATCH --partition=gpu
#SBATCH --output=../slurm_outputs/noble_run/out/%j.out
#SBATCH --error=../slurm_outputs/noble_run/err/%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate noble

cd "$PWD/../src/"

# model settings
model_config=(
  --n_modes 256
  --hidden_channels 24
  --n_layers 12
)

# optimizer settings
optimiser_config=(
  --lr 0.004
  --patience 4
  --scheduler_factor 40
)

# embedding settings
embedding_config=(
  --num_current_embeddings 9
  --type_current_embeddings freq
  --num_hof_model_embeddings 1
  --type_hof_model_embeddings freq
)

# loss settings
loss_config_flags=(
  --train_loss L4
  --train_loss_type rel
)

# training run flags
training_flags=(
  --epochs 300
  --batch_size_train 64
  --batch_size_test 64
  --plot_freq 500
  --save_model False
  --model_save_freq 500
  --print_freq 1
)

path_flags=(
  --data_path PATH_TO_DATA #TODO: Add path to the data for training
  --model_path "$PWD/../training_save_files/noble_models/noble_run/"
  --figure_path "$PWD/../training_save_files/noble_figures/noble_run/"
  --e_features_path "$PWD/../data/e_features/pvalb_689331391_ephys_sim_features.csv"
)

# run training script
python -m training.train_noble \
  --cfg_path "$PWD/../src/training/configs/noble.yaml" \
  "${model_config[@]}" \
  "${optimiser_config[@]}" \
  "${embedding_config[@]}" \
  "${training_flags[@]}" \
  "${loss_config_flags[@]}" \
  "${path_flags[@]}"
