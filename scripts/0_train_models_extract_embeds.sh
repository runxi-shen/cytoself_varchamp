#!/bin/bash

## Modify this path to your actual Conda installation directory:
CONDA_PATH="$HOME/software/anaconda3"

## Source the conda.sh script to initialize Conda in the current shell
## This tells bash how to find and use 'conda activate'
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate cytoself

# Increase file descriptor limits to prevent "too many files open" errors
ulimit -n 65536
ulimit -Sn 65536
ulimit -Hn 65536

# Set environment variables to control parallel processing
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# export NUMBA_NUM_THREADS=4

## 1. Download the images files and their CellProfiler features from the Cell Painting Gallery server
## bash 1_download_cpg_image.sh
## Download the images/CP features from CPG using python scripts is another option but requires extra packages
## python 1_download_image.py
## =========================================================================================================================================================

## 2. Rescale the image intensity and zip them to zarr files for easier processing
python 2_convert_tiff_zarr.py --batch_id "2024_01_23_Batch_7,2024_02_06_Batch_8,2024_12_09_Batch_11,2024_12_09_Batch_12,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"
## =========================================================================================================================================================

## 3. Create the cell-crop imgs for training
# python 3_filter_crop_images.py \
#     --batch_id "2024_01_23_Batch_7,2024_02_06_Batch_8,2024_12_09_Batch_11,2024_12_09_Batch_12,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"
# python 3_filter_cells_by_s2n_fast.py
## =========================================================================================================================================================

## 4. Re-train the Cytoself model using VarChAMP dataset
# python 4_train_model.py --datapath "../inputs/1_model_input/2025_07_B78-1112-1314-1516_clean" --outputpath "../outputs/trained_models" --model_nm "varchamp_080125_clean_data" > training_varchamp_080125_clean_data.log
## =========================================================================================================================================================

## 5. Run analysis on the existing trained model (with the NaN/Inf fix applied)
# python 4_analyze_model.py \
#     --datapath "../inputs/1_model_input/2025_07_B78-1112-1314-1516_clean" \
#     --model_path "../outputs/trained_models/varchamp_080125_clean_data/model_43.pt" \
#     --model_nm "varchamp_080125_clean_data" \
#     > analysis_varchamp_080125_clean_data.log 2>&1
## =========================================================================================================================================================

## 6. Extract embeddings and compute UMAPs
# python 5_get_embeddings.py \
#     --datapath "../inputs/1_model_input/2025_07_B78-1112-1314-1516_clean" \
#     --model_path "../outputs/trained_models/varchamp_080125_clean_data/model_43.pt" \
#     --outputpath "../outputs/trained_models/varchamp_080125_clean_data" \
#     > embeddings_varchamp_080125_clean_data.log 2>&1
## =========================================================================================================================================================
