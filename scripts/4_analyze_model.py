"""
    Run analysis on an already trained cytoself model.
    This script loads a pre-trained model and performs downstream analysis including:
    - Bi-clustering heatmap generation
    - Feature spectrum computation
    - Embedding analysis
"""
import os
import logging
import argparse
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
import matplotlib.pyplot as plt


def analyze_cytoself_model(datapath, model_path, model_nm, data_ch):
    """
    Load pre-trained model and run analysis
    
    Parameters:
    - datapath: Path to the training data
    - model_path: Path to the trained model file (.pt)
    - model_nm: Model name for output paths
    - data_ch: Data channels used during training
    """
    
    # Recreate datamanager with same parameters as training
    datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
    datamanager.const_dataloader(batch_size=32, label_name_position=1)

    print(f"Training size: {datamanager.train_loader.dataset.label.shape}")
    print(f"Test size: {datamanager.test_loader.dataset.label.shape}")

    # Recreate model architecture (must match training)
    model_args = {
        "input_shape": (2, 100, 100),
        "emb_shapes": ((25, 25), (4, 4)),
        "output_shape": (2, 100, 100),
        "fc_output_idx": [2],
        "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
        "num_class": len(datamanager.unique_labels),
        "fc_input_type": "vqvec",
    }
    
    # Create trainer and load the pre-trained model
    train_args = {"lr": 0.0004}  # Only needed for trainer initialization
    model_dir = os.path.dirname(model_path)
    trainer = CytoselfFullTrainer(train_args, homepath=model_dir, model_args=model_args)
    
    # Load the trained model
    print(f"Loading pre-trained model from: {model_path}")
    trainer.load_model(model_path)
    
    # Set trainer to evaluation mode
    trainer.model.eval()
    
    print("Starting analysis from embedding computation...")
    
    # Get test images for reconstruction visualization
    img = next(iter(datamanager.test_loader))["image"].detach().cpu().numpy()
    torch.cuda.empty_cache()
    
    # Generate reconstructed images as sanity check
    print("Generating reconstructed images...")
    reconstructed = trainer.infer_reconstruction(img)

    fig, ax = plt.subplots(2, len(data_ch), figsize=(5 * len(data_ch), 5), squeeze=False)
    for ii, ch in enumerate(data_ch):
        t0 = np.zeros((2 * 100, 5 * 100))
        for i, im in enumerate(img[:10, ii, ...]):
            i0, i1 = np.unravel_index(i, (2, 5))
            t0[i0 * 100 : (i0 + 1) * 100, i1 * 100 : (i1 + 1) * 100] = im
        t1 = np.zeros((2 * 100, 5 * 100))
        for i, im in enumerate(reconstructed[:10, ii, ...]):
            i0, i1 = np.unravel_index(i, (2, 5))
            t1[i0 * 100 : (i0 + 1) * 100, i1 * 100 : (i1 + 1) * 100] = im
        ax[0, ii].imshow(t0, cmap="gray")
        ax[0, ii].axis("off")
        ax[0, ii].set_title("input " + ch)
        ax[1, ii].imshow(t1, cmap="gray")
        ax[1, ii].axis("off")
        ax[1, ii].set_title("output " + ch)

    fig.tight_layout()
    fig.show()
    fig.savefig(Path(trainer.savepath_dict["visualization"]) / "reconstructed_images.png", dpi=300)

    # 3. Analyze embeddings
    print("Initializing analysis object...")
    analysis = AnalysisOpenCell(datamanager, trainer)

    # 3.1 Generate bi-clustering heatmap (this is where the error occurred)
    print("Generating bi-clustering heatmap...")
    analysis.plot_clustermap(num_workers=4)

    # 3.2 Generate feature spectrum
    print("Computing feature spectrum...")
    vqindhist1 = trainer.infer_embeddings(img, "vqindhist1")
    ft_spectrum = analysis.compute_feature_spectrum(vqindhist1)

    x_max = ft_spectrum.shape[1] + 1
    x_ticks = np.arange(0, x_max, 50)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.stairs(ft_spectrum[0], np.arange(x_max), fill=True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Counts")
    ax.set_xlim([0, x_max])
    ax.set_xticks(x_ticks, analysis.feature_spectrum_indices[x_ticks])
    fig.tight_layout()
    fig.show()
    fig.savefig(Path(analysis.savepath_dict["feature_spectra_figures"]) / "feature_spectrum.png", dpi=300)
    
    print("Analysis completed successfully!")


def main():
    data_ch = ["pro", "nuc"]

    parser = argparse.ArgumentParser(description="Analyze pre-trained Cytoself model")
    parser.add_argument("--datapath", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file (.pt)")
    parser.add_argument("--model_nm", type=str, required=True,
                        help="Model name for output organization")
    args = parser.parse_args()

    datapath = args.datapath
    model_path = args.model_path
    model_nm = args.model_nm

    # Setup logging
    log_dir = "../outputs/analysis_logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/cytoself_analysis_{model_nm}_{datetime.today().strftime('%m%d%Y')}.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    overall_start = time.time()
    logging.info(f"Starting analysis of model: {model_path}")

    # Run analysis
    analyze_cytoself_model(datapath, model_path, model_nm, data_ch=data_ch)

    overall_end = time.time()
    elapsed = overall_end - overall_start
    logging.info("Cytoself model analysis finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()