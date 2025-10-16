#!/bin/bash
#SBATCH --job-name=OA_Train                           # Job name
#SBATCH --nodes=1                                     # Number of nodes
#SBATCH --gres=gpu:1                                  # Number of GPUs
#SBATCH --cpus-per-task=8                             # Number of CPU cores per task
#SBATCH --time=1-00:00:00                             # Time limit (Corrected format for 4 days)
#SBATCH --output=logs/train_slurm_mae.out     # Standard output log
#SBATCH --error=logs/train_slurm_mae.err      # Error log
#SBATCH --partition=small_job

# Set up the environment
#source activate ssl_new                              # Activate your conda environment
export CUDA_VISIBLE_DEVICES=0                         # Only requesting 1 GPU with --gres=gpu:1
export PYTHONPATH="/home/joanna/SSLORS/src:$PYTHONPATH"


#EPOCHS=200
#LEARNING_RATE=3e-3 

DEVICES=1                                      # Number of devices for training
NUM_WORKERS=8                                  # Number of data loader workers
MODEL=moco                                     # Model name
TRAIN_BATCH_SIZE=64                            # Training batch size
VAL_BATCH_SIZE=64                              # Validation batch size
LEARNING_RATE=1e-4                             # Learning rate
EPOCHS=100                                     # Number of epochs Hydro uses 800
DATASET_PATH="/mnt/storagecube/joanna/Hydro/"  # Dataset path
SEED=42                                        # Fixed assignment (removed spaces)
# For geo labels: "/home/joanna/SSLORS2/src/utils/train_geo_labels10.csv"
CSV_FILE="/home/joanna/SSLORS2/src/utils/ocean_features/csv_files/ocean_clusters.csv"
LIMIT_FILES=True


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
  --limit_files ${LIMIT_FILES}\
  --csv_file ${CSV_FILE} 

# Optional: Launch TensorBoard
#tensorboard --logdir ./results/trains --port 8009 &
