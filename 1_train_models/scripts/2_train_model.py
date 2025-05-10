"""
    Re-train the cytoself model on the Cell Painting dataset.
"""
import os
import logging
import argparse
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path  # noqa: D100, INP001
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
from cytoself.trainer.utils.plot_history import plot_history_cytoself


def train_cytoself_model(datapath, outputpath, model_nm, data_ch):
    # Define datamanager
    datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
    datamanager.const_dataloader(batch_size=32, label_name_position=1)

    print(f"Training size: {datamanager.train_loader.dataset.label.shape}")
    print(f"Test size: {datamanager.test_loader.dataset.label.shape}")

    model_args = {
        "input_shape": (2, 100, 100),
        "emb_shapes": ((25, 25), (4, 4)),
        "output_shape": (2, 100, 100),
        "fc_output_idx": [2],
        "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
        "num_class": len(datamanager.unique_labels),
        "fc_input_type": "vqvec",
    }
    train_args = {
        "lr": 0.0004,
        "max_epoch": 100,
        "reducelr_patience": 4,
        "reducelr_increment": 0.1,
        "earlystop_patience": 12,
    }
    trainer = CytoselfFullTrainer(train_args, homepath=f"{outputpath}/{model_nm}", model_args=model_args)
    trainer.fit(datamanager, tensorboard_path="tb_logs")

    # 2.1 Generate training history
    plot_history_cytoself(trainer.history, savepath=trainer.savepath_dict["visualization"])

    # 2.2 Compare the reconstructed images as a sanity check
    img = next(iter(datamanager.test_loader))["image"].detach().cpu().numpy()
    torch.cuda.empty_cache()
    reconstructed = trainer.infer_reconstruction(img)

    import matplotlib.pyplot as plt
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
    analysis = AnalysisOpenCell(datamanager, trainer)

    # 3.1 Generate bi-clustering heatmap
    analysis.plot_clustermap(num_workers=4)

    # 3.2 Generate feature spectrum
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


def main():
    data_ch = ["pro", "nuc"]

    parser = argparse.ArgumentParser(description="Training Cytoself model")
    parser.add_argument("--datapath", type=str, 
                        help=f"datapath")
    parser.add_argument("--outputpath", type=str, 
                        help=f"path to store the model")
    parser.add_argument("--model_nm", type=str, help=f"model name")
    args = parser.parse_args()

    datapath = args.datapath
    outputpath = args.outputpath
    model_nm = args.model_nm

    os.makedirs(outputpath, exist_ok=True)

    logging.basicConfig(
        filename=f"../outputs/training_logs/cytoself_model_{model_nm}_{datetime.today().strftime('%m%d%Y')}.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    overall_start = time.time()
    logging.info(f"Training and output cytoself model to {outputpath}/{model_nm}")

    # 2. Train Cytoself model
    train_cytoself_model(datapath, outputpath, model_nm, data_ch=data_ch)

    overall_end = time.time()
    elapsed = overall_end - overall_start
    logging.info("Cytoself model training finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()