#!/usr/bin/env python3
"""
Filter cells by s2n_ratio > 2 and maintain consistency across label, pro, and nuc arrays.
Uses threading for efficient s2n_ratio calculation.
"""

import os
import re
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings


def calculate_s2n_ratio_image(image_2d):
    """Calculate signal-to-noise ratio for a 2D image using 99th/25th percentile."""
    # Flatten the 2D image to 1D for percentile calculation
    pixels = image_2d.flatten()
    
    # Check for invalid values (NaN, inf, etc.)
    if not np.isfinite(pixels).all():
        # Filter out non-finite values
        pixels = pixels[np.isfinite(pixels)]
        if len(pixels) == 0:
            return np.nan
    
    # Suppress numpy warnings for this calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p99 = np.percentile(pixels, 99)
        p25 = np.percentile(pixels, 25)
    
    return p99 / p25 if p25 != 0 else np.inf


def calculate_s2n_ratio_threaded_3d(data_3d, n_threads=None):
    """
    Calculate s2n_ratio for each 2D image in a 3D array using threading.
    
    Parameters:
    - data_3d: 3D numpy array with shape (n_images, height, width)
    - n_threads: number of threads to use (default: 4)
    
    Returns:
    - numpy array of s2n_ratios for each image
    """
    if n_threads is None:
        n_threads = min(4, data_3d.shape[0])  # Use at most 4 threads or number of images
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        s2n_ratios = list(executor.map(calculate_s2n_ratio_image, data_3d))
    
    return np.array(s2n_ratios)


def get_allele_files(dir_path):
    """
    Returns a dict mapping allele -> {file_type: full_path}.
    Expected file pattern: {allele}_{file_type}.npy, where file_type is label, pro, or nuc.
    """
    allele_files = {}
    pattern = re.compile(r'^(?P<allele>.+)_(?P<type>label|pro|nuc)\.npy$')
    for filename in os.listdir(dir_path):
        if filename.endswith('.npy'):
            match = pattern.match(filename)
            if match:
                allele = match.group("allele")
                file_type = match.group("type")
                allele_files.setdefault(allele, {})[file_type] = os.path.join(dir_path, filename)
            else:
                print(f"Warning: {filename} in {dir_path} does not match expected pattern")
    return allele_files


def filter_allele_by_s2n(allele, allele_files, output_dir, s2n_threshold=2.0, n_threads=4):
    """
    Filter cells for a single allele based on protein s2n_ratio threshold.
    Maintains consistency across label, pro, and nuc arrays.
    
    Parameters:
    - allele: allele name
    - allele_files: dict with file paths for this allele
    - output_dir: directory to save filtered files
    - s2n_threshold: minimum s2n_ratio to keep cells
    - n_threads: number of threads for s2n calculation
    
    Returns:
    - tuple: (original_cell_count, filtered_cell_count)
    """
    expected_types = ["label", "pro", "nuc"]
    
    # Check that all required files exist
    missing_files = [ft for ft in expected_types if ft not in allele_files]
    if missing_files:
        print(f"Warning: Allele {allele} missing files: {missing_files}")
        return 0, 0
    
    # Load all arrays
    try:
        label_arr = np.load(allele_files["label"], allow_pickle=True)
        pro_arr = np.load(allele_files["pro"], allow_pickle=True)
        nuc_arr = np.load(allele_files["nuc"], allow_pickle=True)
    except Exception as e:
        print(f"Error loading files for allele {allele}: {e}")
        return 0, 0
    
    # Validate array shapes are consistent
    n_cells = label_arr.shape[0]
    if pro_arr.shape[0] != n_cells or nuc_arr.shape[0] != n_cells:
        print(f"Error: Inconsistent cell counts for allele {allele}")
        print(f"  Label: {label_arr.shape[0]}, Pro: {pro_arr.shape[0]}, Nuc: {nuc_arr.shape[0]}")
        return 0, 0
    
    # Calculate s2n ratios for protein channel
    print(f"Calculating s2n ratios for {allele} ({n_cells} cells)...")
    s2n_ratios = calculate_s2n_ratio_threaded_3d(pro_arr, n_threads=n_threads)
    
    # Filter cells based on s2n threshold
    valid_mask = (np.isfinite(s2n_ratios)) & (s2n_ratios > s2n_threshold)
    n_filtered = np.sum(valid_mask)
    
    if n_filtered == 0:
        print(f"Warning: No cells passed s2n filter for allele {allele}")
        return n_cells, 0
    
    # Apply filter to all arrays
    filtered_label = label_arr[valid_mask]
    filtered_pro = pro_arr[valid_mask]
    filtered_nuc = nuc_arr[valid_mask]
    
    # Save filtered arrays
    for file_type, filtered_arr in [("label", filtered_label), ("pro", filtered_pro), ("nuc", filtered_nuc)]:
        output_filename = f"{allele}_{file_type}.npy"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, filtered_arr)
    
    print(f"  Allele {allele}: {n_cells} -> {n_filtered} cells "
          f"(kept {n_filtered/n_cells*100:.1f}%), "
          f"s2n range: {np.nanmin(s2n_ratios[valid_mask]):.2f}-{np.nanmax(s2n_ratios[valid_mask]):.2f}")
    
    return n_cells, n_filtered


def main():
    # Input and output directories
    input_dir = "../inputs/1_model_input/2025_07_B78-1112-1314-1516"
    output_dir = "../inputs/1_model_input/2025_07_B78-1112-1314-1516_clean"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all allele files
    allele_files_dict = get_allele_files(input_dir)
    all_alleles = sorted(allele_files_dict.keys())
    
    print(f"Found {len(all_alleles)} alleles to process")
    print(f"Output directory: {output_dir}")
    print(f"S2N threshold: 2.0")
    print("-" * 50)
    
    # Process each allele
    total_original = 0
    total_filtered = 0
    
    for allele in tqdm(all_alleles, desc="Processing alleles"):
        original_count, filtered_count = filter_allele_by_s2n(
            allele, 
            allele_files_dict[allele], 
            output_dir,
            s2n_threshold=2.0,
            n_threads=4
        )
        total_original += original_count
        total_filtered += filtered_count
    
    print("-" * 50)
    print(f"Summary:")
    print(f"  Total original cells: {total_original:,}")
    print(f"  Total filtered cells: {total_filtered:,}")
    print(f"  Retention rate: {total_filtered/total_original*100:.1f}%")
    print(f"  Filtered out: {total_original-total_filtered:,} cells")


if __name__ == "__main__":
    main()