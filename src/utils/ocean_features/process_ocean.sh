#!/bin/bash
#SBATCH --job-name=Ocean_CSV          # Job name
#SBATCH --partition=big_job      # Partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --time=5-00:00:00              # Time limit
#SBATCH --output=logs/process_ocean.out  # Standard output log
#SBATCH --error=logs/process_ocean.err   # Error log

# --- Setup Environment ---
#source activate ssl_new

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0             # Adjust for the GPUs you want to use
export PYTHONPATH="/home/joanna/SSLORS/src:$PYTHONPATH"


# --- Run Training ---
srun python -u process_ocea_features.py 

