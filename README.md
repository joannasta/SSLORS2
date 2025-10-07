# Development of Foundation Models for Ocean Remote Sensing

A Master's Thesis submitted to the Technische Universität Berlin | October 2025

## Overview

This repository contains the official code, models, and pre-trained weights for the Master's Thesis titled: Development of Foundation Models for Ocean Remote Sensing.
The project addresses the scarcity of labeled data in marine remote sensing by leveraging Self-Supervised Learning (SSL). Different foundation models were trained using the Hydro dataset containing ocean satellite imagery to learn robust, generalized feature representations. These foundation models are then finetuned on the downstream tasks of bathymetry and marine debris detection.

## Core Contributions

Novel Ocean-Specific SSL Methods: Implementation and evaluation of two novel self-supervised pre-training techniques:
- Ocean Aware
- OceanMAE
  
Comparative Study: Comparing generative and contrastive SSL frameworks:
- MAE
- MoCo
- Geography-Aware
  
Downstream Tasks: Evaluating performance improvements on:
- Bathymetry Regression (pixel-level depth prediction)
- Marine Debris Detection (segmentation)
  
Pre-trained Weights

Model	Backbone	Filename	Link
MAE	ViT	mae.pth	https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models/releases/latest/download/mae_resnet50.pth
OceanMAE	ViT	oceanmae.pth	https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models/releases/latest/download/oceanmae_resnet50.pth
MoCo	ResNet-18	moco.pth	https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models/releases/latest/download/moco_resnet50.pth
Geography_Aware	ResNet-18	geography_aware.pth	https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models/releases/latest/download/geography_aware_resnet50.pth
Ocean_Aware	ResNet-18	ocean_aware.pth	https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models/releases/latest/download/ocean_aware_resnet50.pth


## Datasets

Phase	Dataset	Task Type	Description
Pre-training (Unlabeled)	Hydro [1]	Self-Supervision	100k sampled 256×256 Sentinel‑2 patches containing water from around the globe; used to train the foundation models.
Finetuning (Labeled)	MAGICBathyNet [2]	Bathymetry Regression	Multimodal remote sensing dataset for benchmarking learning‑based bathymetry and pixel‑based classification in shallow waters.
Finetuning (Labeled)	MARIDA [3]	Marine Debris Detection	Marine debris–oriented dataset on Sentinel‑2 images, including various co‑existing sea features.

## Installation and Setup

Environment: Python (PyTorch ecosystem)
BASH
# Clone the repository
git clone https://github.com/Viola-Joanna-Stamer/Ocean-Foundation-Models.git
cd Ocean-Foundation-Models

# Create and activate a conda environment (recommended)
conda create -n ocean-ssl python=3.9
conda activate ocean-ssl

# Install dependencies
pip install -r requirements.txt

### Finetuning Example

BASH
# Example: Finetune the 'mae_ocean' backbone for Marine Debris Detection
python sh finetune_slurm_marida.sh
    --model mae_ocean \
    --pretrained_model ".\results\trains\mae_ocean\version_9\checkpoints\epoch=99-step=2900.ckpt"
    --dataset_path  path\to\marida\dataset
    --learning_rate 1e-5 
    --gpus 1
    
## Quantitative Results

### Domain-specific Best Models (Overview)

Domain	Task	Best Model	Key Metrics	Score
MARIDA	Marine Debris Detection	OceanMAE	IoU / PA / F1	0.600 / 0.750 / 0.700
MAGICBathyNet – Agia Napa	Bathymetry Regression	MAE	MAE / RMSE / STDV	0.431 / 0.571 / 0.517
MAGICBathyNet – Puck Lagoon	Bathymetry Regression	Baseline	MAE / RMSE / STDV	0.493 / 0.907 / 0.874
Note: On Puck Lagoon, the Baseline outperforms the pretrained variants.

### MARIDA — Marine Debris Detection (Leaderboard)

Model (Pretraining)	IoU ↑	PA ↑	F1 ↑
Baseline	0.570	0.690	0.690
MAE	0.480	0.680	0.590
OceanMAE	0.600	0.750	0.700
MoCo	0.340	0.550	0.440
Geography_Aware	0.480	0.630	0.560
Ocean_Aware	0.500	0.660	0.610

### MAGICBathyNet — Agia Napa (Leaderboard)

Model (Pretraining)	MAE ↓	RMSE ↓	STDV ↓
Baseline	0.694	1.068	0.940
MAE	0.431	0.571	0.517
OceanMAE	0.488	0.658	0.564
MoCo	0.514	0.668	0.601
Geography_Aware	0.531	0.689	0.588
Ocean_Aware	0.453	0.589	0.568

### MAGICBathyNet — Puck Lagoon (Leaderboard)

Model (Pretraining)	MAE ↓	RMSE ↓	STDV ↓
Baseline	0.493	0.907	0.874
MAE	0.542	0.783	0.649
OceanMAE	0.775	1.211	0.892
MoCo	0.547	0.886	0.768
Geography_Aware	0.791	1.062	0.929
Ocean_Aware	0.981	1.240	1.058

## Thesis / Code

If you use this code or the pre-trained weights in your research, please cite the Master's Thesis:
BIBTEX
@mastersthesis{stamer2025ocean,
    author  = {Viola-Joanna Stamer},
    title   = {Development of Foundation Models for Ocean Remote Sensing},
    school  = {Technische Universität Berlin},
    year    = {2025},
    month   = {October},
    address = {Berlin, Germany}
}

## References
[1] Corley, I., & Robinson, C. (2024). Hydro Foundation Model [GitHub repository]. https://github.com/isaaccorley/hydro-foundation-model
[2] Agrafiotis, P., Janowski, Ł., Skarlatos, D., & Demir, B. (2024). MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters. In IGARSS 2024 (pp. 249–253). https://doi.org/10.1109/IGARSS53475.2024.10641355
[3] Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., & Karantzalos, K. (2022). MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. PLOS ONE, 17(1), e0262247. https://doi.org/10.1371/journal.pone.0262247
