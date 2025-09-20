#!/usr/bin/env python3
"""
Complete end-to-end pipeline for subcellular localization data integration

This script downloads, merges, and annotates subcellular localization data from:
- Human Protein Atlas (HPA) - downloaded automatically
- OpenCell - manually downloaded from https://opencell.sf.czbiohub.org/download  
- UniProt - retrieved via API using official gene names

Pipeline steps:
1. Download HPA data and verify OpenCell data hash
2. Merge HPA and OpenCell datasets by ENSG ID
3. Annotate with UniProt API using official gene names only
4. Create comprehensive final dataset
5. Clean up intermediate files

Usage:
    python subcellular_data_pipeline.py [--skip-download] [--output-dir DIR]

Output files:
    - final_merged_subcellular_data.csv: Complete integrated dataset
    - uniprot/uniprot_annotations.csv: Raw UniProt annotations 
    - dataset_summary.txt: Pipeline statistics
"""

import os
import sys
import argparse
import requests
import pandas as pd
import time
import hashlib
import pooch
import zipfile
from multiprocessing import Pool, cpu_count
from functools import partial


def download_and_verify_data(output_dir="."):
    """Download HPA data and verify OpenCell data with hash check"""
    print("=== Step 1: Downloading HPA and verifying OpenCell data ===")
    
    # HPA download configuration
    hpa_config = {
        "url": "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip",
        "filename": "subcellular_location.tsv.zip",
        "folder": "proteinatlas",
        "is_zip": True
    }
    
    # OpenCell verification configuration
    opencell_config = {
        "filename": "opencell-localization-annotations.csv",
        "expected_hash": "14cf3536d9d3699460dd34e166d001a34e5430fb551488538b9468107f9f10b6",
        "folder": "opencell",
        "source": "https://opencell.sf.czbiohub.org/download"
    }
    
    downloaded_files = []
    
    # Download HPA data
    print(f"Downloading {hpa_config['filename']}...")
    hpa_folder = os.path.join(output_dir, hpa_config["folder"])
    os.makedirs(hpa_folder, exist_ok=True)
    
    try:
        hpa_file_path = pooch.retrieve(
            url=hpa_config["url"],
            known_hash=None,  # Skip hash for HPA
            path=hpa_folder,
            fname=hpa_config["filename"]
        )
        
        # Extract HPA zip file
        with zipfile.ZipFile(hpa_file_path, 'r') as zip_ref:
            zip_ref.extractall(hpa_folder)
        
        downloaded_files.append(hpa_file_path)
        print(f"HPA data downloaded and extracted to: {hpa_folder}")
        
    except Exception as e:
        print(f"Failed to download HPA data: {e}")
        return None
    
    # Verify OpenCell data
    print(f"Verifying OpenCell file: {opencell_config['filename']}...")
    opencell_folder = os.path.join(output_dir, opencell_config["folder"])
    opencell_file_path = os.path.join(opencell_folder, opencell_config["filename"])
    
    if os.path.exists(opencell_file_path):
        with open(opencell_file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash == opencell_config["expected_hash"]:
            print(f"OpenCell file verified successfully: {opencell_file_path}")
            downloaded_files.append(opencell_file_path)
        else:
            print(f"OpenCell file hash mismatch!")
            print(f"Expected: {opencell_config['expected_hash']}")
            print(f"Got: {file_hash}")
            print(f"Please re-download from {opencell_config['source']}")
            return None
    else:
        print(f"OpenCell file not found: {opencell_file_path}")
        print(f"Please download manually from {opencell_config['source']}")
        return None
    
    print(f"Data verification complete. Files: {len(downloaded_files)}")
    return downloaded_files


def load_and_merge_datasets(output_dir="."):
    """Load HPA and OpenCell data and merge by ENSG ID"""
    print("\\n=== Step 2: Loading and merging HPA and OpenCell datasets ===")
    
    # Find data files
    hpa_file = None
    for file in os.listdir(os.path.join(output_dir, "proteinatlas")):
        if file.endswith('.tsv') and 'subcellular' in file:
            hpa_file = os.path.join(output_dir, "proteinatlas", file)
            break
    
    opencell_file = os.path.join(output_dir, "opencell", "opencell-localization-annotations.csv")
    
    if not hpa_file or not os.path.exists(opencell_file):
        print("Required data files not found!")
        return None
    
    # Load datasets
    print(f"Loading HPA data from: {hpa_file}")
    hpa_df = pd.read_csv(hpa_file, sep='\\t')
    
    print(f"Loading OpenCell data from: {opencell_file}")
    opencell_df = pd.read_csv(opencell_file)
    
    print(f"HPA dataset: {hpa_df.shape}")
    print(f"OpenCell dataset: {opencell_df.shape}")
    
    # Find ENSG ID columns
    hpa_gene_col = None
    opencell_gene_col = None
    
    for col in hpa_df.columns:
        if 'Gene' in col and hpa_df[col].astype(str).str.startswith('ENSG').any():
            hpa_gene_col = col
            break
    
    for col in opencell_df.columns:
        if opencell_df[col].astype(str).str.startswith('ENSG').any():
            opencell_gene_col = col
            break
    
    if not hpa_gene_col or not opencell_gene_col:
        print("Could not find ENSG ID columns for merging")
        return None
    
    print(f"Using HPA gene column: {hpa_gene_col}")
    print(f"Using OpenCell gene column: {opencell_gene_col}")
    
    # Rename columns to add source prefixes
    hpa_df_renamed = hpa_df.copy()
    hpa_df_renamed.columns = [f"HPA_{col}" if col != hpa_gene_col else "ensg_id" for col in hpa_df.columns]
    
    opencell_df_renamed = opencell_df.copy()
    opencell_df_renamed.columns = [f"OpenCell_{col}" if col != opencell_gene_col else "ensg_id" for col in opencell_df.columns]
    
    # Merge datasets
    print("Merging datasets by ENSG ID...")
    merged_df = pd.merge(
        hpa_df_renamed,
        opencell_df_renamed,
        on="ensg_id",
        how="outer"
    )
    
    print(f"Merged dataset: {merged_df.shape}")
    print(f"Total unique genes: {merged_df['ensg_id'].nunique()}")
    
    # Save merged dataset
    merged_file = os.path.join(output_dir, "merged_hpa_opencell.csv")
    merged_df.to_csv(merged_file, index=False)
    print(f"Merged dataset saved to: {merged_file}")
    
    return merged_df


def fetch_uniprot_batch(batch_info):
    """Fetch UniProt data for a batch of ENSG IDs (multiprocessing function)"""
    batch, batch_num, total_batches = batch_info
    
    base_url = "https://rest.uniprot.org/uniprotkb"
    fields = [
        "accession", "id", "gene_names", "protein_name", "organism_name",
        "cc_subcellular_location", "go_c", "go_id", "xref_ensembl"
    ]
    
    # Try xref strategy first
    try:
        gene_query = " OR ".join([f"xref:{ensg_id}" for ensg_id in batch])
        params = {
            "query": f"({gene_query}) AND organism_id:9606",
            "format": "tsv",
            "fields": ",".join(fields),
            "size": str(len(batch) * 3)
        }
        
        response = requests.get(f"{base_url}/search", params=params, timeout=10)
        response.raise_for_status()
        
        if response.text.strip():
            from io import StringIO
            batch_df = pd.read_csv(StringIO(response.text), sep='\\t')
            
            if not batch_df.empty:
                print(f"  Batch {batch_num}/{total_batches}: Retrieved {len(batch_df)} entries")
                return batch_df.to_dict('records')
        
    except Exception as e:
        print(f"  Batch {batch_num}/{total_batches}: Error - {e}")
    
    print(f"  Batch {batch_num}/{total_batches}: No results found")
    return []


def annotate_with_uniprot(merged_df, output_dir=".", batch_size=50, max_workers=None):
    """Annotate genes with UniProt data using multiprocessing"""
    print("\\n=== Step 3: Annotating with UniProt data ===")
    
    # Extract unique ENSG IDs
    ensg_ids = merged_df['ensg_id'].dropna().unique()
    print(f"Annotating {len(ensg_ids)} unique genes using UniProt API...")
    
    # Prepare batches
    batches = []
    total_batches = (len(ensg_ids) + batch_size - 1) // batch_size
    
    for i in range(0, len(ensg_ids), batch_size):
        batch = ensg_ids[i:i+batch_size]
        batch_num = i // batch_size + 1
        batches.append((batch, batch_num, total_batches))
    
    # Set up multiprocessing
    if max_workers is None:
        max_workers = min(8, cpu_count())
    
    print(f"Using {max_workers} workers for {len(batches)} batches...")
    
    # Fetch UniProt data in parallel
    all_uniprot_data = []
    with Pool(max_workers) as pool:
        results = pool.map(fetch_uniprot_batch, batches)
        
        for batch_results in results:
            all_uniprot_data.extend(batch_results)
    
    if all_uniprot_data:
        uniprot_df = pd.DataFrame(all_uniprot_data)
        
        # Save to uniprot directory
        uniprot_dir = os.path.join(output_dir, "uniprot")
        os.makedirs(uniprot_dir, exist_ok=True)
        uniprot_file = os.path.join(uniprot_dir, "uniprot_annotations.csv")
        uniprot_df.to_csv(uniprot_file, index=False)
        
        print(f"UniProt annotations saved to: {uniprot_file}")
        print(f"Total UniProt entries retrieved: {len(uniprot_df)}")
        return uniprot_df
    else:
        print("No UniProt data retrieved")
        return None


def create_final_dataset(merged_df, uniprot_df, output_dir="."):
    """Create final integrated dataset with proper gene name matching"""
    print("\\n=== Step 4: Creating final integrated dataset ===")
    
    if uniprot_df is not None and not uniprot_df.empty:
        # Extract official gene names from UniProt (first name only)
        print("Processing UniProt gene names (using official names only)...")
        uniprot_df['official_gene_name'] = uniprot_df['Gene Names'].apply(
            lambda x: str(x).strip().split()[0] if pd.notna(x) else None
        )
        
        # Show sample processing
        print("Sample UniProt gene name processing:")
        sample = uniprot_df[['Gene Names', 'official_gene_name']].head(5)
        for _, row in sample.iterrows():
            print(f"  '{row['Gene Names']}' -> '{row['official_gene_name']}'")
        
        # Rename UniProt columns to avoid conflicts
        uniprot_cols = {}
        for col in uniprot_df.columns:
            if col != 'official_gene_name':
                uniprot_cols[col] = f"UniProt_{col}"
        uniprot_df_renamed = uniprot_df.rename(columns=uniprot_cols)
        
        # Merge using official gene names
        print("\\nMerging with UniProt data using official gene names...")
        final_df = merged_df.merge(
            uniprot_df_renamed,
            left_on='HPA_Gene name',
            right_on='official_gene_name',
            how='left'
        )
        
        # Calculate match statistics
        total_uniprot_entries = final_df['UniProt_Entry'].notna().sum()
        unique_genes_matched = final_df[final_df['UniProt_Entry'].notna()]['ensg_id'].nunique()
        
        print(f"UniProt integration results:")
        print(f"  - Total UniProt entries matched: {total_uniprot_entries}")
        print(f"  - Unique genes with UniProt data: {unique_genes_matched}/{len(merged_df)} ({unique_genes_matched/len(merged_df)*100:.1f}%)")
        print(f"  - Note: Some genes have multiple UniProt entries (isoforms/variants)")
        
        # Show sample matches
        if total_uniprot_entries > 0:
            print("\\nSample successful matches:")
            matched_sample = final_df[final_df['UniProt_Entry'].notna()]
            for i, (_, row) in enumerate(matched_sample.head(5).iterrows()):
                print(f"  {row['HPA_Gene name']} -> {row['UniProt_Entry']} ({row['UniProt_Gene Names']})")
        
    else:
        print("No UniProt data to integrate")
        final_df = merged_df
        total_uniprot_entries = 0
        unique_genes_matched = 0
    
    # Save final dataset
    final_file = os.path.join(output_dir, "final_merged_subcellular_data.csv")
    final_df.to_csv(final_file, index=False)
    print(f"\\nFinal integrated dataset saved to: {final_file}")
    print(f"Final dataset dimensions: {final_df.shape}")
    
    # Create summary report
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Subcellular Localization Data Integration Summary\\n")
        f.write("=" * 55 + "\\n\\n")
        f.write(f"Total unique genes: {final_df['ensg_id'].nunique()}\\n")
        f.write(f"Final dataset dimensions: {final_df.shape}\\n\\n")
        
        f.write("Data source coverage:\\n")
        hpa_records = final_df.dropna(subset=[col for col in final_df.columns if col.startswith('HPA_')], how='all').shape[0]
        opencell_records = final_df.dropna(subset=[col for col in final_df.columns if col.startswith('OpenCell_')], how='all').shape[0]
        
        f.write(f"  - HPA records: {hpa_records}\\n")
        f.write(f"  - OpenCell records: {opencell_records}\\n")
        f.write(f"  - UniProt entries: {total_uniprot_entries}\\n")
        f.write(f"  - Genes with UniProt data: {unique_genes_matched} ({unique_genes_matched/len(merged_df)*100:.1f}%)\\n\\n")
        
        f.write("Output files:\\n")
        f.write(f"  - Final integrated dataset: {final_file}\\n")
        f.write(f"  - UniProt annotations: uniprot/uniprot_annotations.csv\\n")
        f.write(f"  - This summary: {summary_file}\\n\\n")
        
        f.write("Data sources:\\n")
        f.write("  - HPA: https://www.proteinatlas.org/\\n")
        f.write("  - OpenCell: https://opencell.sf.czbiohub.org/download\\n")
        f.write("  - UniProt: https://www.uniprot.org/\\n")
    
    print(f"Summary report saved to: {summary_file}")
    
    return final_df


def cleanup_files(output_dir="."):
    """Clean up intermediate files"""
    print("\\n=== Step 5: Cleaning up intermediate files ===")
    
    files_to_remove = [
        os.path.join(output_dir, "proteinatlas", "subcellular_location.tsv.zip"),
        os.path.join(output_dir, "merged_hpa_opencell.csv")
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")
    
    # List final files
    print("\\nFinal output files:")
    final_files = [
        "final_merged_subcellular_data.csv",
        "dataset_summary.txt",
        "uniprot/uniprot_annotations.csv"
    ]
    
    for filename in final_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  - {filename} ({size_mb:.1f} MB)")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description="Complete pipeline for subcellular localization data integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python subcellular_data_pipeline.py                    # Full pipeline
  python subcellular_data_pipeline.py --skip-download    # Skip HPA download
  python subcellular_data_pipeline.py --output-dir ./data # Custom output directory
        """
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HPA download if data already exists")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for all files (default: current directory)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Subcellular Localization Data Integration Pipeline")
    print("=" * 55)
    
    try:
        # Step 1: Download/verify data
        if not args.skip_download:
            download_result = download_and_verify_data(output_dir)
            if download_result is None:
                print("Data download/verification failed. Exiting.")
                return 1
        else:
            print("Skipping download step (--skip-download specified)")
        
        # Step 2: Load and merge HPA + OpenCell
        merged_df = load_and_merge_datasets(output_dir)
        if merged_df is None:
            print("Dataset loading/merging failed. Exiting.")
            return 1
        
        # Step 3: Annotate with UniProt
        uniprot_df = annotate_with_uniprot(merged_df, output_dir)
        
        # Step 4: Create final dataset
        final_df = create_final_dataset(merged_df, uniprot_df, output_dir)
        
        # Step 5: Cleanup
        cleanup_files(output_dir)
        
        print("\\n" + "=" * 55)
        print("🎉 Pipeline completed successfully!")
        print("\\nCheck the following files:")
        print("  - final_merged_subcellular_data.csv (main output)")
        print("  - dataset_summary.txt (statistics)")
        print("  - uniprot/uniprot_annotations.csv (raw UniProt data)")
        
        return 0
        
    except Exception as e:
        print(f"\\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())