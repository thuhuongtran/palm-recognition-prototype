import os
import tifffile
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PalmDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (albumentations.Compose): Augmentations to apply.
        """
        self.root_dir = root_dir
        self.files = sorted(
            [os.path.join(root_dir, "session1", f)
             for f in os.listdir(os.path.join(root_dir, "session1"))
             if f.lower().endswith('.tiff')]
            +
            [os.path.join(root_dir, "session2", f)
             for f in os.listdir(os.path.join(root_dir, "session2"))
             if f.lower().endswith('.tiff')]
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        # Load image (grayscale or color; adapt as needed)
        image = tifffile.imread(img_path)

        # Convert single-channel to HxWx1 if needed
        if len(image.shape) == 2:
            # Add channel dimension for consistency
            # shape: (H, W) -> (H, W, 1)
            image = image[:, :, None]

        # Albumentations expects images in HxWxC format
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # now a torch Tensor
        else:
            # Convert NumPy to Tensor if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Dummy label (0) since it's unlabeled
        label = 0
        return image, label


def get_transforms():
    """
    Define your Albumentations pipeline for self-supervised training.
    Include typical augmentations, plus a ToTensorV2 at the end.
    """
    return A.Compose([
        # Example augmentations:
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        # Additional palm-specific transformations if desired...
        # A.GaussNoise(p=0.3),
        # A.IAAPerspective(p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,)),  # for grayscale; adjust if 3-channel
        ToTensorV2()
    ])
