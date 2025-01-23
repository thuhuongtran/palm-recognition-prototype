"""
data_augmentation.py

Applies various data augmentation transformations to TIFF images, including:
- Geometric transformations (rotation, flips, translations)
- Photometric adjustments (brightness, contrast, noise)
- Perspective transforms (shear/warp)
- Random cropping/resizing, blurring, etc.

Saves augmented images to an output directory, preserving original file names
with a suffix or prefix indicating augmentation.
"""

import os
import tifffile
import albumentations as A
import numpy as np


def create_augmentation_pipeline():
    """
    Create an Albumentations augmentation pipeline with various transformations.
    Adjust the parameters as needed.
    """
    transform = A.Compose([
        # Geometric: Rotate +/- 15 degrees
        A.Rotate(limit=15, p=0.5),

        # Geometric: Horizontal or Vertical flip
        A.VerticalFlip(p=0.5),

        # Geometric: Random shift/scale/rotate
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5
        ),

        # Photometric: Random brightness & contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Photometric: Add Gaussian noise
        A.GaussNoise(p=0.3),

        # Perspective / Affine transforms
        A.Perspective(scale=(0.01, 0.05), p=0.3),

        # Random cropping & resizing example
        # Adjust height/width based on your image size
        A.RandomResizedCrop(
            size=[224, 224],
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),

        # Blurring
        A.Blur(blur_limit=3, p=0.3),
    ])
    return transform


def augment_image(image, transform):
    """
    Apply the given Albumentations transform to an image (NumPy array).
    Returns the augmented image (NumPy array).
    """
    augmented = transform(image=image)
    return augmented["image"]


def augment_images_in_directory(
        input_dir,
        output_dir,
        transform,
        num_augmentations_per_image=1
):
    """
    For each TIFF file in input_dir:
      1. Read the image
      2. Generate `num_augmentations_per_image` augmented images
      3. Save them to output_dir with modified filenames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tiff_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith('.tiff')
    ])

    print(f"Found {len(tiff_files)} TIFF files in {input_dir}.")
    print(f"Saving augmented images to {output_dir}\n")

    for i, filename in enumerate(tiff_files, 1):
        file_path = os.path.join(input_dir, filename)

        # Read the original image
        try:
            image = tifffile.imread(file_path)
        except Exception as e:
            print(f"[!] Could not read file: {file_path}, skipping. Error: {e}")
            continue

        # Ensure image is in a valid format (e.g., 2D or 3D array)
        # Albumentations expects HxW (grayscale) or HxWxC (color).
        # If needed, expand dims for single-channel images.
        if len(image.shape) == 2:
            # shape is (height, width), assume single-channel
            pass  # Albumentations can handle this just fine
        elif len(image.shape) == 3:
            # shape is (height, width, channels)
            pass
        else:
            print(f"[!] Unusual image shape {image.shape} for file {filename}, skipping.")
            continue

        # Generate augmented images
        for aug_index in range(num_augmentations_per_image):
            aug_img = augment_image(image, transform)

            # Construct new file name
            # e.g., original: 00001.tiff -> augmented: 00001_aug0.tiff
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{base_name}_aug{aug_index}{ext}"
            out_path = os.path.join(output_dir, new_filename)

            # Save the augmented image
            tifffile.imwrite(out_path, aug_img)

        # Optional: Print progress
        if i % 50 == 0:
            print(f"Processed {i} / {len(tiff_files)} files...")


def main():
    # Paths to your dataset directories
    session1_dir = '../dataset/archive/session1'
    session2_dir = '../dataset/archive/session2'

    # Output directories for augmented images
    session1_aug_dir = '../dataset/augmented/session1'
    session2_aug_dir = '../dataset/augmented/session2'

    # Create the augmentation pipeline
    transform_pipeline = create_augmentation_pipeline()

    # Number of augmented copies per original image
    num_aug = 2  # Adjust as needed

    print("=== Augmenting Session 1 ===")
    augment_images_in_directory(
        input_dir=session1_dir,
        output_dir=session1_aug_dir,
        transform=transform_pipeline,
        num_augmentations_per_image=num_aug
    )

    print("=== Augmenting Session 2 ===")
    augment_images_in_directory(
        input_dir=session2_dir,
        output_dir=session2_aug_dir,
        transform=transform_pipeline,
        num_augmentations_per_image=num_aug
    )

    print("Data augmentation complete!")


if __name__ == "__main__":
    main()
