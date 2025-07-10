#!/bin/bash
#SBATCH --job-name=MOCOGOEO_Train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G 
#SBATCH --output=/home/joanna/SSLORS2/logs/train_mocogeo_ocean_slurm.out 
#SBATCH --error=/home/joanna/SSLORS2/logs/train_mocogeo_ocean_slurm.err  
#SBATCH --partition=small_job 


echo "DEBUG (Shell): Script started."

# Navigate to your project's root directory
echo "DEBUG (Shell): Changing directory to /home/joanna/SSLORS2/"
cd /home/joanna/SSLORS2/
if [ $? -ne 0 ]; then echo "ERROR (Shell): Failed to change directory. Exiting."; exit 1; fi
echo "DEBUG (Shell): Current directory: $(pwd)"

# Load the Miniforge3 initialization script
echo "DEBUG (Shell): Sourcing /home/joanna/miniforge3/etc/profile.d/conda.sh"
source /home/joanna/miniforge3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then echo "ERROR (Shell): Failed to source conda.sh. Exiting."; exit 1; fi
echo "DEBUG (Shell): Conda.sh sourced successfully."

# Set environment variables
echo "DEBUG (Shell): Setting environment variables."
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/joanna/SSLORS/src:$PYTHONPATH"

# Define variables
DEVICES=1
NUM_WORKERS=8
MODEL=moco-geo-ocean
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=64
LEARNING_RATE=3e-3
EPOCHS=200
DATASET_PATH="/data/joanna/Hydro/"
SEED=42
NUM_CLUSTERS=10
OCEAN=TRUE

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

echo "Training completed."