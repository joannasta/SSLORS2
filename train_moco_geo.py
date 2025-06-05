#!/bin/bash
#SBATCH --job-name=MAE_Train # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --partition=rsim_member # Partition name
#SBATCH --time=4-00:00:00 # Time limit
#SBATCH --output=logs/train_slurm_geo.out # Standard output log
#SBATCH --error=logs/train_slurm_geo.err # Error log

# Set up the environment
#source activate ssl_new # Activate your conda environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 # GPUs to be visible
export PYTHONPATH="/home/joanna/SSLORS/src:$PYTHONPATH"

# Define variables
DEVICES=1 # Number of devices for training
NUM_WORKERS=8 # Number of data loader workers
MODEL=moco-geo # Model name
TRAIN_BATCH_SIZE=64 # Training batch size
VAL_BATCH_SIZE=64 # Validation batch size
LEARNING_RATE=3e-3 # Learning rate
EPOCHS=200 # Number of epochs Hydro uses 800
DATASET_PATH="/faststorage/joanna/Hydro/raw_data" # Dataset path
SEED=42 # Random seed
NUM_CLUSTERS=10 # Number of clusters for clustering

# Run the training script
srun python -u train.py \
--accelerator gpu \
--devices ${DEVICES} \
--train-batch-size ${TRAIN_BATCH_SIZE} \
--val-batch-size ${VAL_BATCH_SIZE} \
--num-workers ${NUM_WORKERS} \
--learning-rate ${LEARNING_RATE} \
--dataset ${DATASET_PATH} \
--model ${MODEL} \
--epochs ${EPOCHS} \
--seed ${SEED} \

# Optional: Launch TensorBoard
#tensorboard --logdir ./results/trains --port 8009 &

# Completion message
echo "Training completed." >> /home/joanna/Joanna/logs/train_slurm.out

