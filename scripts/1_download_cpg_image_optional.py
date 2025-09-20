# """Download images and metadata for batch 7.

# The process involves:
# 1. Querying the cpg index
# 2. Downloading files

# """

# # noqa: INP001
# # Imports
# import argparse
# import logging
# import time
# from pathlib import Path
# import polars as pl
# from cpgdata.utils import download_s3_files, parallel


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Image batch to download")
#     parser.add_argument("--batch_id", type=str, 
#                         help=f"Batch ID")
#     args = parser.parse_args()

#     # Override defaults with command line args if provided
#     batch_id = args.batch_id

#     # Configure logging to output to a file
#     logging.basicConfig(
#         filename=f"1_download_image_{batch_id}.log",
#         filemode="w",
#         level=logging.DEBUG,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )
#     overall_start = time.time()
#     logging.info("Downloading started")


#     """Download images and metadata.

#     This function uses the cpg index to locate all DNA and GFP images and Image.csv files for image batch.

#     """
#     logging.info(f"Starting to downloading raw images from cell-painting gallery...")
#     index_dir = Path("../cp_gallery_index_files")
#     index_files = list(index_dir.glob("*.parquet"))

#     # Download images
#     index_df = pl.scan_parquet(index_files)
#     index_df = (
#         index_df
#         .filter(pl.col("dataset_id").eq("cpg0020-varchamp"))
#         .filter(pl.col("batch_id") == batch_id) ## "2024_01_23_Batch_7"
#         .filter(pl.col("leaf_node").str.contains(".tiff"))
#         .filter((pl.col("leaf_node").str.contains("-ch1")) | (pl.col("leaf_node").str.contains("-ch2")))
#         .select("key")
#         .collect()
#     )
    
#     download_keys = list(index_df.to_dict()["key"])
#     parallel(download_keys, download_s3_files, ["cellpainting-gallery",
#                                                 Path("../cellpainting_gallery/tiff_images")],
#                                                 jobs=20)
#     logging.info(f"tiff images downloaded to ../cellpainting_gallery/tiff_images.")

#     # Download Image.csv
#     index_df = pl.scan_parquet(index_files)

#     index_df = (
#         index_df
#         .filter(pl.col("dataset_id").eq("cpg0020-varchamp"))
#         .filter(pl.col("batch_id") == batch_id) ## "2024_01_23_Batch_7"
#         .filter(pl.col("leaf_node").str.contains("Image.csv"))
#         .filter(pl.col("key").str.contains("assaydev"))
#         .select("key")
#         .collect()
#     )

#     download_keys = list(index_df.to_dict()["key"])
#     parallel(download_keys, download_s3_files, ["cellpainting-gallery",
#                                                 Path("../cellpainting_gallery/image_csv")],
#                                                 jobs=20)
#     logging.info(f"cellprofiler csv features downloaded to ../cellpainting_gallery/image_csv.")

#     # index_df = (
#     #     index_df
#     #     .filter(pl.col("dataset_id").eq("cpg0020-varchamp"))
#     #     .filter(pl.col("batch_id") == batch_id) ## "2024_01_23_Batch_7"
#     #     .filter(pl.col("leaf_node").str.contains("Cells.csv"))
#     #     .filter(pl.col("key").str.contains("assaydev"))
#     #     .select("key")
#     #     .collect()
#     # )

#     # download_keys = list(index_df.to_dict()["key"])
#     # parallel(download_keys, download_s3_files, ["cellpainting-gallery",
#     #                                             Path("../cellpainting_gallery/cells_csv")],
#     #                                             jobs=20)
#     # logging.info(f"cellprofiler csv features downloaded to ../cellpainting_gallery/cells_csv.")


#     overall_end = time.time()
#     elapsed = overall_end - overall_start
#     logging.info("Downloading finished in %.2f seconds", elapsed)


# if __name__ == "__main__":
#     main()
