"""
    Get the embeddings of the images in the VarChAMP dataset.
"""
import os
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

### Set parameters
datapath = '../inputs/1_model_input/2024_05_B78-1314-1516'
outputpath = "../outputs/trained_models/varchamp_050725"
model_nm = "model_43.pt"
data_ch = ['pro', 'nuc']
device="cuda:0"

# model_path = f"{outputpath}/{model_nm}"
# datamanager = DataManagerOpenCell(datapath, data_ch, fov_col=None)
# datamanager.const_dataloader(batch_size=32, label_name_position=1)

# model_args = {
#     "input_shape": (2, 100, 100),
#     "emb_shapes": ((25, 25), (4, 4)),
#     "output_shape": (2, 100, 100),
#     "fc_output_idx": [2],
#     "vq_args": {"num_embeddings": 512, "embedding_dim": 64},
#     "num_class": len(datamanager.unique_labels),
#     "fc_input_type": "vqvec",
# }
# train_args = {
#     "lr": 0.0004,
#     "max_epoch": 100,
#     "reducelr_patience": 4,
#     "reducelr_increment": 0.1,
#     "earlystop_patience": 12,
# }

# trainer = CytoselfFullTrainer(train_args, homepath=outputpath, model_args=model_args)
# trainer.load_model(model_path)

"""
    Infer embeddings
"""
# ## infer embeddings
# test_embeddings = trainer.infer_embeddings(datamanager.test_loader)

# if isinstance(test_embeddings, tuple) and len(test_embeddings) > 1:
#     test_embeddings = test_embeddings[0]

# savepath_embeddings = trainer.savepath_dict["embeddings"]
# np.save(join(savepath_embeddings, 'embeddings_testdata.npy'), test_embeddings)

# # infer embeddings for training data 
# train_embeddings = trainer.infer_embeddings(datamanager.train_loader)
# if isinstance(train_embeddings, tuple) and len(train_embeddings) > 1:
#     train_embeddings = train_embeddings[0]

# savepath_embeddings = trainer.savepath_dict["embeddings"]
# np.save(join(savepath_embeddings, 'embeddings_traindata.npy'), train_embeddings)

# # infer embeddings for training data 
# valid_embeddings = trainer.infer_embeddings(datamanager.val_loader)
# if isinstance(valid_embeddings, tuple) and len(valid_embeddings) > 1:
#     valid_embeddings = valid_embeddings[0]

# savepath_embeddings = trainer.savepath_dict["embeddings"]
# np.save(join(savepath_embeddings, 'embeddings_valdata.npy'), valid_embeddings)

# # save labels by data
# test_labels = datamanager.test_loader.dataset.label
# np.save(join(savepath_embeddings, 'labels_testdata.npy'), test_labels)

# train_labels = datamanager.train_loader.dataset.label
# np.save(join(savepath_embeddings, 'labels_traindata.npy'), train_labels)

# val_labels = datamanager.val_loader.dataset.label
# np.save(join(savepath_embeddings, 'labels_valdata.npy'), val_labels)


## plot embeddings by umap
n_neighbors=15
min_dist=0.1
metric='euclidean'
verbose=True

# trainer_embeddings = np.load(join(f"{outputpath}/embeddings", 'embeddings_traindata.npy'))
# val_embeddings = np.load(join(f"{outputpath}/embeddings", 'embeddings_valdata.npy'))
# test_embeddings = np.load(join(f"{outputpath}/embeddings", 'embeddings_testdata.npy'))

# train_labels = np.load(join(f"{outputpath}/embeddings", 'labels_traindata.npy'), allow_pickle=True)
# val_labels = np.load(join(f"{outputpath}/embeddings", 'labels_valdata.npy'), allow_pickle=True)
# test_labels = np.load(join(f"{outputpath}/embeddings", 'labels_testdata.npy'), allow_pickle=True)

os.makedirs("../outputs/analysis/umap_data", exist_ok=True)
reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, verbose=verbose, random_state=42)
umap_data = reducer.fit_transform(test_embeddings.reshape(test_embeddings.shape[0], -1))
umap_df = pl.DataFrame({
    "Metadata_Protein": list(test_labels[:, 0]),
    "Metadata_CellID": list(test_labels[:, 1]),
    "Comp1": umap_data[:, 0],
    "Comp2": umap_data[:, 1]
})
umap_df.write_parquet(f"../outputs/analysis/umap_data/umap_comps_testdata.parquet")

reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, verbose=verbose, random_state=42)
umap_data = reducer.fit_transform(val_embeddings.reshape(val_embeddings.shape[0], -1))
umap_df = pl.DataFrame({
    "Metadata_Protein": list(val_labels[:, 0]),
    "Metadata_CellID": list(val_labels[:, 1]),
    "Comp1": umap_data[:, 0],
    "Comp2": umap_data[:, 1]
})
umap_df.write_parquet(f"../outputs/analysis/umap_data/umap_comps_valdata.parquet")