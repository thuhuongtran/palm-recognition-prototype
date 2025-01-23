import os
import shutil
import random


def split_files(file_list, train_ratio=0.8, seed=42):
    """
    Splits a list of files into train and test subsets.
    train_ratio defines the fraction of data to go into train.
    Returns two lists: train_files, test_files
    """
    random.seed(seed)
    random.shuffle(file_list)
    train_size = int(len(file_list) * train_ratio)
    train_files = file_list[:train_size]
    test_files = file_list[train_size:]
    return train_files, test_files


def create_split_dirs(base_dir):
    """
    Creates train/session1, train/session2, test/session1, test/session2
    inside base_dir if they don't already exist.
    """
    paths = [
        os.path.join(base_dir, 'train', 'session1'),
        os.path.join(base_dir, 'train', 'session2'),
        os.path.join(base_dir, 'test', 'session1'),
        os.path.join(base_dir, 'test', 'session2'),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


def split_and_organize(session_dir, dest_base_dir, train_ratio=0.8, seed=42):
    """
    - Takes a session_dir (e.g., session1 source).
    - Splits the TIFF files into train/test.
    - Copies each subset to the appropriate directory under dest_base_dir.
    """
    # Identify which session (session1 or session2) from the directory name
    session_name = os.path.basename(session_dir)  # e.g. "session1" or "session2"

    # Gather TIFF files
    all_files = sorted(
        f for f in os.listdir(session_dir)
        if f.lower().endswith('.tiff')
    )

    # Full paths
    all_file_paths = [os.path.join(session_dir, f) for f in all_files]

    # Split
    train_files, test_files = split_files(
        file_list=all_file_paths,
        train_ratio=train_ratio,
        seed=seed
    )

    # Destination directories
    train_dest = os.path.join(dest_base_dir, 'train', session_name)
    test_dest = os.path.join(dest_base_dir, 'test', session_name)

    # Copy files
    for fpath in train_files:
        shutil.copy(fpath, train_dest)
    for fpath in test_files:
        shutil.copy(fpath, test_dest)

    print(f"\nSession '{session_name}':")
    print(f"  Total files: {len(all_file_paths)}")
    print(f"  Train split: {len(train_files)} -> {train_dest}")
    print(f"  Test split : {len(test_files)} -> {test_dest}")


def main():
    # Source directories (after cleaning)
    session1_dir = "../dataset/archive/session1"
    session2_dir = "../dataset/archive/session2"

    # Destination base directory
    # e.g. ../dataset/split_data/
    # it will create train/session1, train/session2, test/session1, test/session2 under this directory
    dest_base_dir = "../dataset/split_data"

    # Create the split subdirectories
    create_split_dirs(dest_base_dir)

    # Perform the split for each session
    split_and_organize(session1_dir, dest_base_dir, train_ratio=0.8, seed=42)
    split_and_organize(session2_dir, dest_base_dir, train_ratio=0.8, seed=42)

    print("\nDataset splitting completed!")


if __name__ == "__main__":
    main()
