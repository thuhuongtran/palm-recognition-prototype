from torchvision import transforms


def get_palmprint_augmentations(image_size):
    augmentation_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8), # Grayscale Jitter (Brightness/Contrast) - applied with probability 0.8
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.5),  # Gaussian Blur - applied with probability 0.5
        # transforms.RandomHorizontalFlip(p=0.5), # Optional: Horizontal Flip - experiment with/without
    ]
    return transforms.Compose(augmentation_list)