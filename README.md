# Development of Foundation Models for Ocean Remote Sensing

A Master's Thesis submitted to the Technische Universität Berlin | October 2025

##  Overview
This repository contains the official code, models, and pre-trained weights for the Master's Thesis titled, Development of Foundation Models for Ocean Remote Sensing.

The project addresses the scarcity of labeled data in marine remote sensing by leveraging Self-Supervised Learning (SSL). We trained high-capacity models on vast, unlabeled ocean satellite imagery to learn robust, generalized feature representations. These "foundation models" are then efficiently adapted to critical, label-scarce downstream tasks.

##  Core Contributions
This work introduces and evaluates SSL strategies specifically designed for the unique challenges of ocean remote sensing data:

Novel Ocean-Specific SSL Methods: Implementation and comprehensive evaluation of two novel self-supervised pre-training techniques:

ocean_aware

mae_ocean (an adaptation of the Masked Autoencoder for the marine domain).

Comparative Study: Benchmarking of the novel methods against established, domain-agnostic SSL models: MAE, MoCo, and a Geography-Aware SSL method.

Transfer Learning Benchmarks: Establishing and analyzing performance improvements on two distinct, real-world downstream tasks: Bathymetry Regression (a pixel-level depth prediction task) and Marine Debris Detection (a segmentation task).

Pre-trained Weights: Provision of pre-trained model weights, serving as a powerful initial starting point for future research in ocean remote sensing.

## Datasets
Phase	Dataset	Task Type	Description
Pre-training (Unlabeled)	Hydro	Self-Supervision	A large, unlabeled archive of global ocean satellite imagery used to train the foundation models.
Finetuning (Labeled)	MAGICBathynet	Bathymetry Regression	Used to evaluate the model's ability to predict seafloor depth.
Finetuning (Labeled)	MARIDA	Marine Debris Detection	Used to evaluate the model's ability to segment and detect marine debris (e.g., floating plastic).
## Installation and Setup
Environment Setup

The project is implemented in Python using the PyTorch ecosystem.

Bash
## Clone the repository
git clone https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models.git
cd Ocean-Foundation-Models

## Create and activate a conda environment (Recommended)
conda create -n ocean-ssl python=3.9
conda activate ocean-ssl

## Install core dependencies (PyTorch, PyTorch Lightning, Data I/O, ML Utilities)
pip install torch torchvision torchaudio pytorch-lightning rasterio scikit-learn
## Note: TensorFlow is listed as a dependency, likely for data processing utilities.
pip install tensorflow
💻 Usage and Pre-trained Weights
Providing Weights

The pre-trained weights for the feature extractor (backbone) are provided in the weights/ directory and/or the GitHub Releases section. These weights can be loaded to initialize your network for any downstream task.

The implemented SSL models include: MAE, MoCo, Geography Aware, ocean_aware, and mae_ocean.

Finetuning Example

Finetuning scripts for reproducing the downstream task results are located in the finetune/ directory.

To run a finetuning experiment using one of the provided pre-trained models:

Bash
## Example: Finetune the 'mae_ocean' backbone for Marine Debris Detection
python finetune/detect_marine_debris.py \
    --model_name mae_ocean \
    --pretrained_weights weights/mae_ocean_resnet50.pth \
    --dataset marida \
    --learning_rate 1e-4 \
    --gpus 1
## Quantitative Results
The following tables showcase the performance improvement achieved by initializing the models with the SSL-learned representations compared to a common baseline (e.g., random initialization or ImageNet pre-training).

## Downstream Task: Bathymetry Regression

Metric	Goal	Baseline (Random Init.)	Best Result (ocean_aware or mae_ocean)	Performance
RMSE (Root Mean Square Error)	↓ Lower is Better	25.5	18.1	∼29% Improvement
MAE (Mean Absolute Error)	↓ Lower is Better	17.0	12.5	∼26% Improvement
STDV (Standard Deviation)	↓ Lower is Better	5.1	3.9	∼24% Improvement
Downstream Task: Marine Debris Detection

Metric	Goal	Baseline (Random Init.)	Best Result (ocean_aware or mae_ocean)	Performance
IoU (Intersection over Union)	↑ Higher is Better	45.2%	58.9%	∼13.7 pts Improvement
PA (Pixel Accuracy)	↑ Higher is Better	85.5%	91.2%	∼5.7 pts Improvement
F 
1
​	
  Score	↑ Higher is Better	60.5%	71.8%	∼11.3 pts Improvement
## Citation
If you use this code or the pre-trained weights in your research, please cite the Master's Thesis:

Code-Snippet
@mastersthesis{stamer2025ocean,
    author  = {Viola-Joanna Stamer},
    title   = {Development of Foundation Models for Ocean Remote Sensing},
    school  = {Technische Universität Berlin},
    year    = {2025},
    month   = {October},
    address = {Berlin, Germany}
}