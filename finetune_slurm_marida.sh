#!/bin/bash
#SBATCH --job-name=MAE_FineTune           # Job name
#SBATCH --partition=thesis_student        # Partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --time=11:00:00                   # Time limit
#SBATCH --output=logs/finetune_slurm_marida.out  # Standard output log
#SBATCH --error=logs/finetune_slurm_marida.err   # Error log

# --- Setup Environment ---
#source activate ssl_new

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0             # Adjust for the GPUs you want to use
export PYTHONPATH="/home/joanna/SSLORS/src:$PYTHONPATH"

# --- Define Training Parameters ---
DEVICES=1                                 # Number of devices for training
NUM_WORKERS=32                             # Number of data loader workers
MODEL=mae                                 # Model name
TRAIN_BATCH_SIZE=16                     # Training batch size
VAL_BATCH_SIZE=16                     # Validation batch size
LEARNING_RATE=1e-5   #1e-6                  # Learning rate
EPOCHS=200                               # Number of epochs
PRETRAINED_MODEL="./results/trains/training_logs/11-channels/checkpoints/epoch=32-step=54747.ckpt" # Path to pretrained model
DATASET_PATH="/faststorage/joanna/marida/MARIDA" # Dataset path
SEED=42                                   # Seed for reproducibility

# --- Run Training ---
srun python -u finetune_marida.py \
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
  --seed ${SEED}
# --- Launch TensorBoard (optional) ---
#tensorboard --logdir ./results/finetuning_marida_mae --port 8009 &

# Completion message
echo "Fine-tuning completed." >> logs/finetune_slurm_marida.out
