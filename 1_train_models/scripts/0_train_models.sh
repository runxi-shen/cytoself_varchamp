#!/bin/bash

## Modify this path to your actual Conda installation directory:
CONDA_PATH="$HOME/software/anaconda3"

## Source the conda.sh script to initialize Conda in the current shell
## This tells bash how to find and use 'conda activate'
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate cytoself

## Create the imgs for training
# python 1_filter_crop_images.py --batch_id "2024_01_23_Batch_7,2024_02_06_Batch_8,2024_12_09_Batch_11,2024_12_09_Batch_12,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"

python 2_train_model.py --datapath "../inputs/1_model_input/2024_05_B78-1314-1516" --outputpath "../outputs/trained_models" --model_nm "varchamp_050725" > training_varchamp_050725.log
