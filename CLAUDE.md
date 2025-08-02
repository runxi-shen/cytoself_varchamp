# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a specialized implementation of cytoself (a self-supervised deep learning platform for protein subcellular localization) adapted for the VarChAMP dataset. The project retrains cytoself models on VarChAMP cell painting data to learn protein localization patterns.

## Commands

### Installation and Setup
```bash
# Create and activate conda environment
conda create -y -n cytoself python=3.9
conda activate cytoself

# Install package in development mode
pip install -e .

# Install development dependencies (for developers)
pip install -r requirements/development.txt
pre-commit install
```

### Testing
```bash
# Run all tests
pytest

# Run tests with slow tests included
pytest --runslow

# Run specific test file
pytest cytoself/test_util/test_parameters.py

# Run tests with coverage (if configured)
pytest --cov=cytoself
```

### Code Quality
```bash
# Format code with black
black .

# Run linting with flake8
flake8

# Sort imports with isort
isort .

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Data Preparation (VarChAMP workflow)
```bash
# Download cell painting images from S3 gallery
cd 0_data_prep/scripts
./0_download_cp_image.sh

# Run full data preparation pipeline
./1_run_data_prep.sh
```

### Model Training
```bash
# Filter and crop images for training
cd 1_train_models/scripts
python 1_filter_crop_images.py --batch_id "batch1,batch2,..."

# Train cytoself model
python 2_train_model.py --datapath "../inputs/..." --outputpath "../outputs/..." --model_nm "model_name"

# Run full training pipeline with conda environment
./4_train_models.sh
```

### Analysis
```bash
# Generate embeddings from trained model
python 3_get_embeddings.py

# Run notebooks for analysis (use Jupyter)
jupyter notebook 1_train_models/notebooks/
```

## Project Structure

### Core Architecture

The project follows the original cytoself architecture with VarChAMP-specific customizations:

- **cytoself/**: Core library with modular components
  - **trainer/**: Different trainer classes (CytoselfFullTrainer, CytoselfLiteTrainer, VanillaAETrainer)
  - **datamanager/**: Data loading and preprocessing (OpenCell format compatibility)
  - **components/**: Neural network building blocks (encoders, decoders, VQ layers)
  - **analysis/**: Tools for UMAP visualization, clustering analysis, and feature interpretation

- **0_data_prep/**: VarChAMP data preprocessing pipeline
  - Downloads images from cell painting gallery
  - Converts TIFF to Zarr format for efficient access
  - Implements quality control metrics

- **1_train_models/**: Model training and analysis
  - **scripts/**: Training pipeline scripts
  - **notebooks/**: Jupyter notebooks for exploration and analysis
  - **pooled_rare/**: Special handling for pooled rare variant data

- **2_phenom_beta/**: Integration with Phenom Beta API for additional analysis

### Key Model Types

1. **CytoselfFullTrainer**: Full implementation with VQ-VAE and fully connected layers
2. **CytoselfLiteTrainer**: Lightweight version for faster training
3. **VanillaAETrainer**: Standard autoencoder baseline

### Data Flow

1. **Input**: Cell painting images (protein + nucleus channels) from VarChAMP dataset
2. **Preprocessing**: Filter cells, crop to 100x100 pixels, quality control
3. **Training**: Self-supervised learning using protein identity as labels
4. **Output**: Learned embeddings capturing protein localization patterns
5. **Analysis**: UMAP visualization, clustering, feature interpretation

## VarChAMP-Specific Considerations

- Uses batch processing for large-scale VarChAMP data (batches 7-19)
- Implements additional QC metrics specific to cell painting data
- Handles zarr format for efficient large image storage
- Includes pooled rare variant processing pipeline
- Integrates with cell painting gallery data structure

## Model Configuration

Default model arguments for VarChAMP data:
```python
model_args = {
    'input_shape': (2, 100, 100),  # protein + nucleus channels
    'emb_shapes': ((25, 25), (4, 4)),
    'output_shape': (2, 100, 100),
    'fc_output_idx': [2],
    'vq_args': {'num_embeddings': 512, 'embedding_dim': 64},
    'num_class': len(datamanager.unique_labels),
    'fc_input_type': 'vqvec',
}
```

Training typically uses batch_size=32, learning rate=0.0004, with early stopping and learning rate reduction on plateau.

## Testing Notes

- Tests use `@pytest.mark.slow` for computationally expensive tests
- Fixtures generate dummy data for testing without requiring full datasets  
- Test configuration in `conftest.py` handles temporary directories and model setup
- Use `--runslow` flag to include slow tests in test runs