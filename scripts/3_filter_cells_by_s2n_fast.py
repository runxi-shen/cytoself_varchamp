#!/usr/bin/env python3
"""
Faster version: Filter cells by s2n_ratio > 2 with batch processing and better progress tracking.
Uses threading for efficient s2n_ratio calculation and parallel file processing.
"""

import os
import re
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from multiprocessing import cpu_count


def calculate_s2n_ratio_batch(pro_data, batch_size=1000):
    """
    Calculate s2n_ratio for protein data in batches to save memory and improve speed.
    
    Parameters:
    - pro_data: 3D numpy array with shape (n_images, height, width)
    - batch_size: number of images to process at once
    
    Returns:
    - numpy array of s2n_ratios for each image
    """
    n_images = pro_data.shape[0]
    s2n_ratios = np.zeros(n_images)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        for i in range(0, n_images, batch_size):
            end_idx = min(i + batch_size, n_images)
            batch = pro_data[i:end_idx]
            
            # Vectorized calculation for the batch
            batch_flat = batch.reshape(batch.shape[0], -1)
            
            # Calculate percentiles for all images in batch at once
            p99 = np.percentile(batch_flat, 99, axis=1)
            p25 = np.percentile(batch_flat, 25, axis=1)
            
            # Calculate ratios with safe division
            batch_ratios = np.divide(p99, p25, out=np.full_like(p99, np.inf), where=(p25 != 0))
            s2n_ratios[i:end_idx] = batch_ratios
    
    return s2n_ratios


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
    return allele_files


def process_single_allele(args):
    """Process a single allele - designed for parallel execution."""
    allele, allele_files, output_dir, s2n_threshold = args
    
    expected_types = ["label", "pro", "nuc"]
    
    # Check that all required files exist
    missing_files = [ft for ft in expected_types if ft not in allele_files]
    if missing_files:
        return allele, 0, 0, f"Missing files: {missing_files}"
    
    try:
        # Load all arrays
        label_arr = np.load(allele_files["label"], allow_pickle=True)
        pro_arr = np.load(allele_files["pro"], allow_pickle=True)
        nuc_arr = np.load(allele_files["nuc"], allow_pickle=True)
        
        # Validate array shapes are consistent
        n_cells = label_arr.shape[0]
        if pro_arr.shape[0] != n_cells or nuc_arr.shape[0] != n_cells:
            return allele, 0, 0, f"Inconsistent cell counts: Label:{label_arr.shape[0]}, Pro:{pro_arr.shape[0]}, Nuc:{nuc_arr.shape[0]}"
        
        # Calculate s2n ratios for protein channel using batch processing
        s2n_ratios = calculate_s2n_ratio_batch(pro_arr, batch_size=1000)
        
        # Filter cells based on s2n threshold
        valid_mask = (np.isfinite(s2n_ratios)) & (s2n_ratios > s2n_threshold)
        n_filtered = np.sum(valid_mask)
        
        if n_filtered == 0:
            return allele, n_cells, 0, "No cells passed s2n filter"
        
        # Apply filter to all arrays
        filtered_label = label_arr[valid_mask]
        filtered_pro = pro_arr[valid_mask]
        filtered_nuc = nuc_arr[valid_mask]
        
        # Save filtered arrays
        for file_type, filtered_arr in [("label", filtered_label), ("pro", filtered_pro), ("nuc", filtered_nuc)]:
            output_filename = f"{allele}_{file_type}.npy"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, filtered_arr)
        
        # Calculate stats for reporting
        valid_s2n = s2n_ratios[valid_mask]
        s2n_min = np.nanmin(valid_s2n)
        s2n_max = np.nanmax(valid_s2n)
        
        return allele, n_cells, n_filtered, f"s2n range: {s2n_min:.2f}-{s2n_max:.2f}"
        
    except Exception as e:
        return allele, 0, 0, f"Error: {str(e)}"


def main():
    # Input and output directories
    input_dir = "../inputs/1_model_input/2025_07_B78-1112-1314-1516"
    output_dir = "../inputs/1_model_input/2025_07_B78-1112-1314-1516_clean"
    s2n_threshold = 2.0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all allele files
    allele_files_dict = get_allele_files(input_dir)
    all_alleles = sorted(allele_files_dict.keys())
    
    # Use fewer workers to avoid memory issues with large arrays
    n_workers = min(4, cpu_count())  # Limit to 4 workers to prevent memory exhaustion
    
    print(f"Found {len(all_alleles)} alleles to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"S2N threshold: {s2n_threshold}")
    print(f"Using parallel processing with {n_workers} workers")
    print("-" * 80)
    
    # Prepare arguments for parallel processing
    process_args = [
        (allele, allele_files_dict[allele], output_dir, s2n_threshold)
        for allele in all_alleles
    ]
    
    # Process alleles in parallel
    total_original = 0
    total_filtered = 0
    error_count = 0
    
    # Use ProcessPoolExecutor for CPU-intensive numpy operations
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks and get futures
        future_to_allele = {
            executor.submit(process_single_allele, args): args[0] 
            for args in process_args
        }
        
        # Process completed tasks as they finish
        with tqdm(total=len(all_alleles), desc="Processing alleles") as pbar:
            from concurrent.futures import as_completed
            for future in as_completed(future_to_allele):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per allele
                    allele, original_count, filtered_count, message = result
                    
                    if "Error" in message or original_count == 0:
                        print(f"FAILED {allele}: {message}")
                        error_count += 1
                    else:
                        retention_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
                        print(f"SUCCESS {allele}: {original_count:,} -> {filtered_count:,} cells "
                              f"({retention_rate:.1f}%), {message}")
                    
                    total_original += original_count
                    total_filtered += filtered_count
                    
                except Exception as e:
                    allele = future_to_allele[future]
                    print(f"FAILED {allele}: Exception during processing: {str(e)}")
                    error_count += 1
                
                pbar.update(1)
    
    print("-" * 80)
    print(f"Summary:")
    print(f"  Total alleles processed: {len(all_alleles)}")
    print(f"  Successful: {len(all_alleles) - error_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total original cells: {total_original:,}")
    print(f"  Total filtered cells: {total_filtered:,}")
    if total_original > 0:
        print(f"  Overall retention rate: {total_filtered/total_original*100:.1f}%")
        print(f"  Cells filtered out: {total_original-total_filtered:,}")
    
    print(f"\nFiltered data saved to: {output_dir}")


if __name__ == "__main__":
    main()