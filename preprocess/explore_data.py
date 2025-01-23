import os
import tifffile
import matplotlib.pyplot as plt


# ---------------------------
# 1. Helper Functions
# ---------------------------

def list_tiff_files(directory):
    """Return a sorted list of all TIFF files in the given directory."""
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith('.tiff')
    ])


def parse_numeric_ids(filenames):
    """
    Convert filenames (e.g., '00001.tiff') to numeric IDs (e.g., 1).
    Returns a list of valid IDs and a list of files that couldn't be parsed.
    """
    numeric_ids = []
    invalid_filenames = []

    for filename in filenames:
        name_only = os.path.splitext(filename)[0]  # remove .tiff extension
        try:
            numeric_id = int(name_only)
            numeric_ids.append(numeric_id)
        except ValueError:
            invalid_filenames.append(filename)

    return numeric_ids, invalid_filenames


def check_missing_ids(numeric_ids, start_id, end_id):
    """
    Check for missing IDs in the range [start_id, end_id].
    Returns a list of any missing IDs.
    """
    expected_range = set(range(start_id, end_id + 1))
    existing_ids = set(numeric_ids)
    missing_ids = list(expected_range - existing_ids)
    missing_ids.sort()  # sort for consistent reporting
    return missing_ids


def is_corrupted(file_path):
    """
    Attempt to read a TIFF file.
    Returns True if corrupted, False if valid.
    """
    try:
        _ = tifffile.imread(file_path)
        return False
    except Exception:
        return True


def find_corrupted_files(directory, filenames):
    """
    For each file in filenames, test if it is corrupted.
    Returns a list of corrupted file paths.
    """
    corrupted = []
    for f in filenames:
        file_path = os.path.join(directory, f)
        if is_corrupted(file_path):
            corrupted.append(file_path)
    return corrupted


def show_sample_image(directory, filenames, corrupted_files):
    """
    Display basic information and a plot of the first non-corrupted image
    in the given list of filenames.
    """
    if not filenames:
        print("No files to display in this session.")
        return

    first_file = filenames[0]
    first_path = os.path.join(directory, first_file)

    if first_path in corrupted_files:
        print(f"Sample file '{first_file}' is corrupted, cannot visualize.")
        return

    image = tifffile.imread(first_path)
    print("\nSample File Info:")
    print("  - Name        :", first_file)
    print("  - Shape       :", image.shape)
    print("  - Dtype       :", image.dtype)
    print(f"  - Value Range : [{image.min()}, {image.max()}]")

    plt.imshow(image, cmap='gray')
    plt.title(f"Sample Image: {first_file}")
    plt.axis('off')
    plt.show()


# ---------------------------
# 2. Main Processing
# ---------------------------

# Paths to your dataset directories
session1_dir = '../dataset/archive/session1'
session2_dir = '../dataset/archive/session2'

# (A) LIST TIFF FILES
session1_files = list_tiff_files(session1_dir)
session2_files = list_tiff_files(session2_dir)

print("Number of TIFF files in session1:", len(session1_files))
print("Number of TIFF files in session2:", len(session2_files))

# (B) PARSE NUMERIC IDS + CHECK MISSING FILES
# Adjust START_ID/END_ID based on your dataset expectations
START_ID, END_ID = 1, 6000

# Session 1
session1_ids, session1_invalid = parse_numeric_ids(session1_files)
session1_missing = check_missing_ids(session1_ids, START_ID, END_ID)

# Session 2
session2_ids, session2_invalid = parse_numeric_ids(session2_files)
session2_missing = check_missing_ids(session2_ids, START_ID, END_ID)

# Report missing and invalid filenames
if session1_missing:
    print(f"\nMissing IDs in Session1 (count: {len(session1_missing)}): {session1_missing[:10]} ...")
else:
    print("\nNo missing file IDs detected in Session1.")

if session2_missing:
    print(f"\nMissing IDs in Session2 (count: {len(session2_missing)}): {session2_missing[:10]} ...")
else:
    print("\nNo missing file IDs detected in Session2.")

if session1_invalid:
    print("\nInvalid filenames in Session1 (non-numeric):", session1_invalid)
if session2_invalid:
    print("\nInvalid filenames in Session2 (non-numeric):", session2_invalid)

# (C) CHECK FOR CORRUPTED FILES
session1_corrupted = find_corrupted_files(session1_dir, session1_files)
session2_corrupted = find_corrupted_files(session2_dir, session2_files)

if session1_corrupted:
    print("\nCorrupted files in Session1:")
    for cf in session1_corrupted:
        print("  -", cf)

if session2_corrupted:
    print("\nCorrupted files in Session2:")
    for cf in session2_corrupted:
        print("  -", cf)

if not session1_corrupted and not session2_corrupted:
    print("\nNo corrupted files detected in either session.")

# (D) CHECK FOR IMBALANCE
num_session1 = len(session1_files)
num_session2 = len(session2_files)

if num_session1 == 0 or num_session2 == 0:
    print("\nCannot determine imbalance because one session has 0 files.")
else:
    ratio = num_session1 / num_session2 if num_session2 != 0 else 0
    print(f"\nSession1 : Session2 Ratio = {ratio:.2f}")
    # Example heuristic: if one session has <50% or >200% of the other
    if ratio < 0.5 or ratio > 2.0:
        print("Warning: The dataset might be imbalanced between sessions based on file counts.")
    else:
        print("No severe imbalance between sessions based on file counts.")

# (E) VISUALIZE A SAMPLE IMAGE
print("\n--- SESSION 1 SAMPLE ---")
show_sample_image(session1_dir, session1_files, session1_corrupted)

print("\n--- SESSION 2 SAMPLE ---")
show_sample_image(session2_dir, session2_files, session2_corrupted)
