#!/bin/bash
#SBATCH --job-name=Moco_MO_FineTune           # Job name
#SBATCH --partition=small_job      # Partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --time=1-00:00:00              # Time limit
#SBATCH --output=logs/finetune_slurm_mbn_mae_ocean_puck_lagoon.out  # Standard output log
#SBATCH --error=logs/finetune_slurm_mbn_mae_ocean_puck_lagoon.err   # Error log

# --- Setup Environment ---
#source activate ssl_new

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0             # Adjust for the GPUs you want to use
export PYTHONPATH="/home/joanna/SSLORS/src:$PYTHONPATH"

# --- Define Training Parameters ---
DEVICES=1                                 # Number of devices for training
NUM_WORKERS=32                              # Number of data loader workers
MODEL=mae_ocean                             # Model name
TRAIN_BATCH_SIZE=1                    # Training batch size
VAL_BATCH_SIZE=1                     # Validation batch size
LEARNING_RATE=1e-4 #1e-5                  # Learning rate
EPOCHS=10                              # Number of epochs

PRETRAINED_MODEL="./results/trains/mae_ocean/version_6/checkpoints/epoch=99-step=9900.ckpt"
#"./results/trains/moco-geo-ocean/version_5/checkpoints/epoch=199-step=16800.ckpt"
#"./results/trains/geo_aware/version_9/checkpoints/epoch=199-step=19800.ckpt"
#"./results/trains/moco/version_55/checkpoints/epoch=99-step=9900.ckpt"
#"./results/trains/mae/version_23/checkpoints/epoch=99-step=10000.ckpt"
#"./results/trains/moco-geo-ocean/version_5/checkpoints/epoch=199-step=16800.ckpt"
#"./results/trains/mae/version_23/checkpoints/epoch=99-step=10000.ckpt"
#"./results/trains/geo_aware/version_9/checkpoints/epoch=199-step=19800.ckpt"



#"./results/trains/training_logs/3-channels/checkpoints/epoch=99-step=132700.ckpt" 
#"./results/trains/moco-geo/version_28/checkpoints/epoch=199-step=331600.ckpt"
#"./results/trains/moco-geo-ocean/version_5/checkpoints/epoch=199-step=16800.ckpt"
#"./results/trains/moco/version_38/checkpoints/epoch=31-step=53056.ckpt"
#"./results/trains/moco-geo-ocean/version_5/checkpoints/epoch=199-step=16800.ckpt"
#"./results/trains/moco-geo/version_28/checkpoints/epoch=199-step=331600.ckpt"
#"./results/trains/moco/version_38/checkpoints/epoch=31-step=53056.ckpt"
#"./results/trains/moco-geo/version_27/checkpoints/epoch=199-step=331600.ckpt"
#"./results/trains/training_logs/3-channels/checkpoints/epoch=99-step=132700.ckpt" 
#"./results/trains/moco/version_38/checkpoints/epoch=31-step=53056.ckpt"
#"./results/trains/moco-geo/version_28/checkpoints/epoch=199-step=331600.ckpt"
DATASET_PATH="/mnt/storagecube/joanna/MagicBathyNet/"  # Dataset path
SEED=42                                   # Seed for reproducibility
LOCATION=puck_lagoon

# --- Run Training ---
srun python -u finetune_mbn.py \
  --weights ${PRETRAINED_MODEL} \
  --data_dir ${DATASET_PATH} \
  --output_dir ./results/inference \
  --accelerator gpu \
  --devices ${DEVICES} \
  --train-batch-size ${TRAIN_BATCH_SIZE} \
  --val-batch-size ${VAL_BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --learning-rate ${LEARNING_RATE} \
  --model ${MODEL} \
  --epochs ${EPOCHS} \
  --seed ${SEED} \
  --location ${LOCATION}
# --- Launch TensorBoard (optional) ---
#tensorboard --logdir ./results/trains --port 8009 &

# Completion message
echo "Fine-tuning completed." >> logs/finetune_slurm_mbn.out
