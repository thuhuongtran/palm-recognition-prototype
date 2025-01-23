"""
data_cleaning.py

Demonstrates:
1. Validation of images (detect/remove corrupted files).
2. Cross-checking metadata consistency with images (placeholder).
3. Removing duplicate images to avoid data leakage.
"""

import os
import hashlib
import tifffile

from preprocess.explore_data import is_corrupted


def remove_corrupted_images(directory, remove_files=False):
    """
    Scan a directory for corrupted TIFF files.
    If remove_files=True, deletes corrupted files from disk.
    Returns a list of corrupted file paths.
    """
    tiff_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith('.tiff')
    ])

    corrupted_files = []

    for f in tiff_files:
        file_path = os.path.join(directory, f)
        if is_corrupted(file_path):
            corrupted_files.append(file_path)

    if corrupted_files:
        print(f"\n[!] Found {len(corrupted_files)} corrupted files in {directory}:")
        for cf in corrupted_files:
            print("  -", cf)

        if remove_files:
            for cf in corrupted_files:
                os.remove(cf)
            print("[!] Corrupted files removed from disk.")
    else:
        print(f"\n[+] No corrupted files found in {directory}.")

    return corrupted_files


def cross_check_metadata(directory, metadata_path=None):
    """
    Placeholder function to demonstrate metadata checking.
    For example, if you have a CSV with a 'filename' column, you can:
      1. Read the CSV.
      2. Compare the list of TIFF filenames in this directory vs. CSV entries.
    Returns any discrepancies found (missing_in_dir, missing_in_metadata).
    """
    if metadata_path is None:
        print(f"\nNo metadata provided for {directory}; skipping consistency check.")
        return [], []

    # Example logic (requires pandas):
    # import pandas as pd
    # df = pd.read_csv(metadata_path)
    # expected_files = set(df['filename'])  # 'filename' column in CSV
    #
    # actual_files = set([
    #     f for f in os.listdir(directory)
    #     if f.lower().endswith('.tiff')
    # ])
    #
    # missing_in_dir = expected_files - actual_files
    # missing_in_metadata = actual_files - expected_files
    #
    # # Print or return these sets
    # return missing_in_dir, missing_in_metadata

    print(f"\n[!] cross_check_metadata() is not implemented. Provide logic as needed.")
    return [], []


def find_duplicate_images(directory):
    """
    Identify duplicate TIFF files based on file content (using a hash).
    Returns a dict: {hash_value: [file_paths]} where len(file_paths) > 1 indicates duplicates.
    """
    tiff_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith('.tiff')
    ])

    hash_dict = {}

    for filename in tiff_files:
        file_path = os.path.join(directory, filename)
        try:
            image_data = tifffile.imread(file_path)
        except Exception:
            # If file is corrupted, skip or handle differently
            continue

        # Compute a hash of the raw bytes in the image
        img_hash = hashlib.md5(image_data.tobytes()).hexdigest()

        if img_hash not in hash_dict:
            hash_dict[img_hash] = []
        hash_dict[img_hash].append(file_path)

    return hash_dict


def remove_duplicates(directory, duplicates_dict):
    """
    Remove all but the first file in each list of duplicates from disk.
    """
    removed_files = []

    for img_hash, file_list in duplicates_dict.items():
        if len(file_list) > 1:
            # Keep the first file, remove the rest
            file_list.sort()  # For consistent behavior, sort by filename
            for file_path in file_list[1:]:
                os.remove(file_path)
                removed_files.append(file_path)

    return removed_files


def clean_directory(directory, metadata_path=None, remove_corrupted=False, remove_dupes=False):
    """
    High-level function to clean a directory by:
      1. Removing (or reporting) corrupted files.
      2. Checking metadata consistency (placeholder).
      3. Removing (or reporting) duplicate files.
    """
    print(f"\n--- Cleaning directory: {directory} ---")

    # 1) Remove/Report corrupted images
    corrupted_files = remove_corrupted_images(directory, remove_files=remove_corrupted)

    # 2) Cross-check metadata consistency (if metadata_path is provided)
    missing_in_dir, missing_in_metadata = cross_check_metadata(directory, metadata_path)

    if missing_in_dir or missing_in_metadata:
        print("[!] Metadata discrepancies detected.")
        print("  - Missing in directory:", missing_in_dir)
        print("  - Missing in metadata :", missing_in_metadata)

    # 3) Find duplicates
    dup_dict = find_duplicate_images(directory)
    duplicates_found = [v for v in dup_dict.values() if len(v) > 1]
    total_duplicates = sum(len(v) - 1 for v in duplicates_found)  # minus 1 each group’s unique image

    if duplicates_found:
        print(f"\n[!] Found {len(duplicates_found)} groups of duplicates (total {total_duplicates} duplicate files).")
        for group in duplicates_found:
            print("  -", group)

        if remove_dupes:
            removed = remove_duplicates(directory, dup_dict)
            print(f"[!] Removed {len(removed)} duplicate files.")
    else:
        print("\n[+] No duplicate images found.")

    print(f"\n--- Finished cleaning directory: {directory} ---\n")


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == '__main__':
    # Paths to your dataset directories
    session1_dir = '../dataset/archive/session1'
    session2_dir = '../dataset/archive/session2'

    # Clean Session1
    clean_directory(
        directory=session1_dir,
        metadata_path=None,  # set to actual path if you have a CSV/JSON file
        remove_corrupted=False,  # set True if you want to auto-remove corrupted files
        remove_dupes=False  # set True if you want to auto-remove duplicates
    )

    # Clean Session2
    clean_directory(
        directory=session2_dir,
        metadata_path=None,
        remove_corrupted=False,
        remove_dupes=False
    )
