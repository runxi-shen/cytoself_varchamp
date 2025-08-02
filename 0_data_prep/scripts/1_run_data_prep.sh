#!/bin/bash

## Modify this path to your actual Conda installation directory:
CONDA_PATH="$HOME/software/anaconda3"

## Source the conda.sh script to initialize Conda in the current shell
## This tells bash how to find and use 'conda activate'
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate cytoself

## Download the images files and their CellProfiler features from the Cell Painting Gallery server
## bash 1_download_cp_image.sh

## Rescale the image intensity and zip them to zarr files for easier processing
python 2_convert_tiff_zarr.py --batch_id "2024_01_23_Batch_7,2024_02_06_Batch_8,2024_12_09_Batch_11,2024_12_09_Batch_12,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"