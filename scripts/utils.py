import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def channel_to_cmap(channel):
    if channel == "pro":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#65fe08"])
    elif channel == "nuc":
        cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#0000FF"])
    # elif channel == "Mito":
    #     cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#FF0000"]) 
    # elif channel == "AGP":
    #     cmap = mpl.colors.LinearSegmentedColormap.from_list("green_cmap", ["#000","#FFFF00"]) 
    else:
        cmap = "gray"
    return cmap


def viz_cell_crop(img_arr, cell_comp, max_intensity=.99, ax=None, axis_off=True):
    cmap = channel_to_cmap(cell_comp)
    if ax is None:
        plt.imshow(img_arr, vmin=0, vmax=np.percentile(img_arr, max_intensity*100), cmap=cmap)
        if axis_off:
            plt.axis('off')  # Turn off axis labels
        plt.show()
    else:
        ax.imshow(img_arr, vmin=0, vmax=np.percentile(img_arr, max_intensity*100), cmap=cmap)
        if axis_off:
            ax.axis('off')
        plot_label = f"s2n ratio:\n{np.percentile(img_arr, 99) / np.percentile(img_arr, 25):.3f}"
        ax.text(0.05, 0.95, plot_label, color='white', fontsize=9,
                verticalalignment='top', horizontalalignment='left', 
                transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.2, linewidth=1))


def visualize_cell_crop_pro(allele, npy_dir="../data/interim/3_model_input/2025_07_B78-1112-1314-1516_clean", 
                           cell_num=10, random_seed=42, cell_ids=[], nuc=True):
    """
        Visualize allele's cell crops:
        # visualize_cell_crop_pro("ABCD1-Arg389Gly")
        # visualize_cell_crop_pro("ABCD1-Arg389Gly", 
        cell_ids=["2025_01_27_B13A7A8P1_T1_E01_871_16","2025_01_27_B13A7A8P1_T1_E01_871_2"], nuc=False)
    """
    # Load data
    ex_meta = np.load(os.path.join(npy_dir, f"{allele}_label.npy"), allow_pickle=True)
    ex_nuc = np.load(os.path.join(npy_dir, f"{allele}_nuc.npy"), allow_pickle=True)
    ex_prot = np.load(os.path.join(npy_dir, f"{allele}_pro.npy"), allow_pickle=True)
    
    # Get cell indices
    if cell_ids:
        # Find indices where cell_ids match ex_meta[:,1]
        cell_indices = [i for i, meta in enumerate(ex_meta) if meta[1] in cell_ids]
        if not cell_indices:
            print(f"No matching cells found for cell_ids: {cell_ids}")
            return
    else:
        # Random selection
        replace = ex_meta.shape[0] < cell_num
        np.random.seed(random_seed)
        cell_indices = np.random.choice(np.arange(ex_meta.shape[0]), 
                                        min(cell_num, ex_meta.shape[0]), replace=replace)
    
    # Extract data for selected cells
    selected_meta = ex_meta[cell_indices]
    selected_nuc = ex_nuc[cell_indices]
    selected_prot = ex_prot[cell_indices]
    num_cells = len(cell_indices)
    
    def create_title(meta_row):
        """Helper function to create consistent titles"""
        cell_id = meta_row[1]
        if "_T" in cell_id:
            parts = cell_id.split("_T")
            return f"{parts[0]}\nT{parts[1]}"
        return cell_id
    
    if nuc:
        # Two-row layout: nucleus on top, protein on bottom
        fig, axes = plt.subplots(2, num_cells, figsize=(num_cells*2, 6))
        
        # Handle single cell case
        if num_cells == 1:
            axes = axes.reshape(2, 1)
        
        for idx in range(num_cells):
            # Plot nucleus
            viz_cell_crop(selected_nuc[idx], "nuc", ax=axes[0, idx])
            title_base = create_title(selected_meta[idx])
            axes[0, idx].set_title(f"{title_base}|nuc", fontsize=7)
            
            # Plot protein
            viz_cell_crop(selected_prot[idx], "pro", ax=axes[1, idx])
            axes[1, idx].set_title(f"{title_base}|pro", fontsize=7)
        
        fig.suptitle(allele)
        fig.subplots_adjust(wspace=.02, hspace=-.45, top=1.05)
    
    else:
        # Single row layout
        fig, axes = plt.subplots(1, num_cells, figsize=(num_cells*2, 3))
        # Handle single cell case
        if num_cells == 1:
            axes = [axes]
        for idx in range(num_cells):
            # Plot protein only
            viz_cell_crop(selected_prot[idx], "pro", ax=axes[idx])
            title_base = create_title(selected_meta[idx])
            axes[idx].set_title(f"{title_base}|pro", fontsize=7)
        
        fig.suptitle(allele)
        fig.subplots_adjust(wspace=.02, top=.9)

