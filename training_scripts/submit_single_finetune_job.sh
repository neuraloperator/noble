#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -J "SAG_FT"
#SBATCH --partition=gpu
#SBATCH --output=../slurm_outputs/noble_finetune_sag_amplitude/out/%j.out
#SBATCH --error=../slurm_outputs/noble_finetune_sag_amplitude/err/%j.err

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
  --optimizer_name "adamw"
  --lr 5e-4
  --scheduler ReduceLROnPlateauTrain
  --scheduler_factor 40
  --patience 12
)

# embedding settings
embedding_config=(
  --sine_embeddings_freq 9
  --scale_sine_embeddings freq
  --hof_model_embeddings 1
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
  --plot_freq 25
  --save_model False
  --print_freq 1
  --feature_loss '{"sag_amplitude": 25}'
  --custom_prefix 'Finetune_SagAmplitude_Weight25'
)

# path flags
path_flags=(
  --data_path PATH_TO_DATA #TODO: Add path to the data for training
  --model_path "$PWD/../training_save_files/noble_models/noble_finetune/"
  --figure_path "$PWD/../training_save_files/noble_figures/noble_finetune/"
  --e_features_path "$PWD/../data/e_features/pvalb_689331391_ephys_sim_features.csv"
)

# finetune run flags
finetune_flags=(
  --pretrained_model_path "$PWD/../noble_models/FNO_nmodes-256_in-23_out-1_nlayers-12_projectionratio-4_hc-24_AmpEmbeddings-FreqScaledNeRF-nfreq-9_HoFEmbeddings-FreqScaledNeRF-nfreq-1_bestepoch-296.pth"
)

# run training script
python -m training.train_noble_finetune \
  --cfg_path "$PWD/../src/training/configs/noble.yaml" \
  "${model_config[@]}" \
  "${optimiser_config[@]}" \
  "${embedding_config[@]}" \
  "${training_flags[@]}" \
  "${loss_config_flags[@]}" \
  "${finetune_flags[@]}" \
  "${path_flags[@]}"
