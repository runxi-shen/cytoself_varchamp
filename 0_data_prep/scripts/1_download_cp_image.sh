#!/bin/bash

AWS_IMG_PATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/images"
AWS_ANALYSIS_PATH="s3://cellpainting-gallery/cpg0020-varchamp/broad/workspace/analysis"
HOMEPATH="../inputs/varchamp_cellpainting_gallery"

## To download all of the sources in the JUMP dataset
# batches=`aws s3 ls --no-sign-request "$BASEPATH/backend/" | awk '{print substr($2, 1, length($2)-1)}'`

## Batch names
# batches="2024_01_23_Batch_7 2024_02_06_Batch_8"
# batches="2024_12_09_Batch_11 2024_12_09_Batch_12"
# batches="2025_01_27_Batch_13 2025_01_28_Batch_14"
# batches="2025_03_17_Batch_15 2025_03_17_Batch_16"

# "2024_01_23_Batch_7,2024_02_06_Batch_8,2025_01_27_Batch_13,2025_01_28_Batch_14,2025_03_17_Batch_15,2025_03_17_Batch_16"

for batch_id in $batches;
do
    aws s3 sync --no-sign-request "$AWS_IMG_PATH/$batch_id/images" $HOMEPATH/$batch_id/images
    aws s3 sync --no-sign-request \
        "$AWS_ANALYSIS_PATH/$batch_id" \
        "$HOMEPATH/$batch_id/analysis" \
        --exclude "*" \
        --include "**/Cells.csv" \
        --include "**/Cytoplasm.csv" \
        --include "**/Nuclei.csv" \
        --include "**/Image.csv" \
        # --recursive \
        # --dry-run
done