"""
    Get the embeddings of the images in the VarChAMP dataset.
    Updated to work with the newly trained model: varchamp_080125_clean_data
"""
import os
import argparse
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import polars as pl
import umap.umap_ as umap
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
from cytoself.trainer.utils.plot_history import plot_history_cytoself
from cytoself.analysis.utils.cluster_score import calculate_cluster_centrosize


def get_embeddings(datapath, model_path, outputpath, data_ch=['pro', 'nuc']):
    """
    Load trained model and generate embeddings for all data splits
    """
    print(f"Loading model from: {model_path}")
    print(f"Data path: {datapath}")
    print(f"Output path: {outputpath}")
    
    # Create datamanager
    datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
    datamanager.const_dataloader(batch_size=32, label_name_position=1)

    print(f"Training size: {datamanager.train_loader.dataset.label.shape}")
    print(f"Validation size: {datamanager.val_loader.dataset.label.shape}")
    print(f"Test size: {datamanager.test_loader.dataset.label.shape}")

    # Create model with same architecture as training
    model_args = {
        "input_shape": (2, 100, 100),
        "emb_shapes": ((25, 25), (4, 4)),
        "output_shape": (2, 100, 100),
        "fc_output_idx": [2],
        "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
        "num_class": len(datamanager.unique_labels),
        "fc_input_type": "vqvec",
    }
    train_args = {"lr": 0.0004}  # Only needed for trainer initialization

    trainer = CytoselfFullTrainer(train_args, homepath=outputpath, model_args=model_args)
    trainer.load_model(model_path)
    trainer.model.eval()  # Set to evaluation mode
    
    return datamanager, trainer


def extract_embeddings(datamanager, trainer):
    """
    Extract embeddings for all data splits
    """
    print("Extracting embeddings...")
    
    # Create embeddings directory
    savepath_embeddings = trainer.savepath_dict["embeddings"]
    os.makedirs(savepath_embeddings, exist_ok=True)
    
    # Extract test embeddings
    print("Extracting test embeddings...")
    test_embeddings = trainer.infer_embeddings(datamanager.test_loader)
    if isinstance(test_embeddings, tuple) and len(test_embeddings) > 1:
        test_embeddings = test_embeddings[0]
    np.save(join(savepath_embeddings, 'embeddings_testdata.npy'), test_embeddings)
    
    # Extract training embeddings 
    print("Extracting training embeddings...")
    train_embeddings = trainer.infer_embeddings(datamanager.train_loader)
    if isinstance(train_embeddings, tuple) and len(train_embeddings) > 1:
        train_embeddings = train_embeddings[0]
    np.save(join(savepath_embeddings, 'embeddings_traindata.npy'), train_embeddings)

    # Extract validation embeddings 
    print("Extracting validation embeddings...")
    valid_embeddings = trainer.infer_embeddings(datamanager.val_loader)
    if isinstance(valid_embeddings, tuple) and len(valid_embeddings) > 1:
        valid_embeddings = valid_embeddings[0]
    np.save(join(savepath_embeddings, 'embeddings_valdata.npy'), valid_embeddings)

    # Save labels for each split
    print("Saving labels...")
    test_labels = datamanager.test_loader.dataset.label
    np.save(join(savepath_embeddings, 'labels_testdata.npy'), test_labels)

    train_labels = datamanager.train_loader.dataset.label
    np.save(join(savepath_embeddings, 'labels_traindata.npy'), train_labels)

    val_labels = datamanager.val_loader.dataset.label
    np.save(join(savepath_embeddings, 'labels_valdata.npy'), val_labels)
    
    print(f"Embeddings and labels saved to: {savepath_embeddings}")
    return test_embeddings, train_embeddings, valid_embeddings, test_labels, train_labels, val_labels


def compute_umaps(outputpath, test_embeddings, train_embeddings, valid_embeddings, 
                  test_labels, train_labels, val_labels):
    """
    Compute UMAP embeddings and save as parquet files
    """
    print("Computing UMAP embeddings...")
    
    # UMAP parameters
    n_neighbors = 15
    min_dist = 0.1
    metric = 'euclidean'
    verbose = True

    # Create output directory for UMAP data
    umap_output_dir = "../outputs/analysis/umap_data"
    os.makedirs(umap_output_dir, exist_ok=True)

    # Compute UMAP for test data
    print("Computing UMAP for test data...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                       metric=metric, verbose=verbose, random_state=42)
    umap_data = reducer.fit_transform(test_embeddings.reshape(test_embeddings.shape[0], -1))
    umap_df = pl.DataFrame({
        "Metadata_Protein": list(test_labels[:, 0]),
        "Metadata_CellID": list(test_labels[:, 1]),
        "Comp1": umap_data[:, 0],
        "Comp2": umap_data[:, 1]
    })
    umap_df.write_parquet(f"{umap_output_dir}/umap_comps_testdata.parquet")

    # Compute UMAP for validation data
    print("Computing UMAP for validation data...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                       metric=metric, verbose=verbose, random_state=42)
    umap_data = reducer.fit_transform(valid_embeddings.reshape(valid_embeddings.shape[0], -1))
    umap_df = pl.DataFrame({
        "Metadata_Protein": list(val_labels[:, 0]),
        "Metadata_CellID": list(val_labels[:, 1]),
        "Comp1": umap_data[:, 0],
        "Comp2": umap_data[:, 1]
    })
    umap_df.write_parquet(f"{umap_output_dir}/umap_comps_valdata.parquet")
    
    # Compute UMAP for training data (optional - can be large)
    print("Computing UMAP for training data...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                       metric=metric, verbose=verbose, random_state=42)
    umap_data = reducer.fit_transform(train_embeddings.reshape(train_embeddings.shape[0], -1))
    umap_df = pl.DataFrame({
        "Metadata_Protein": list(train_labels[:, 0]),
        "Metadata_CellID": list(train_labels[:, 1]),
        "Comp1": umap_data[:, 0],
        "Comp2": umap_data[:, 1]
    })
    umap_df.write_parquet(f"{umap_output_dir}/umap_comps_traindata.parquet")
    
    print(f"UMAP data saved to: {umap_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and compute UMAPs from trained model")
    parser.add_argument("--datapath", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file (.pt)")
    parser.add_argument("--outputpath", type=str, required=True,
                        help="Output directory (model directory)")
    args = parser.parse_args()

    # Load model and data
    datamanager, trainer = get_embeddings(args.datapath, args.model_path, args.outputpath)
    
    # Extract embeddings
    test_embeddings, train_embeddings, valid_embeddings, test_labels, train_labels, val_labels = \
        extract_embeddings(datamanager, trainer)
    
    # Compute UMAPs
    compute_umaps(args.outputpath, test_embeddings, train_embeddings, valid_embeddings,
                  test_labels, train_labels, val_labels)
    
    print("Embedding extraction and UMAP computation completed!")


if __name__ == "__main__":
    main()

# Legacy code (all replaced with functions above - commented out for reference):
# Original hardcoded parameters - now passed as arguments:
# datapath = '../inputs/1_model_input/2024_05_B78-1314-1516'
# outputpath = "../outputs/trained_models/varchamp_050725"
# model_nm = "model_43.pt"
# data_ch = ['pro', 'nuc']

# Original embedding extraction code - now in extract_embeddings() function
# Original UMAP computation code - now in compute_umaps() function