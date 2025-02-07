import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from augmentations import get_palmprint_augmentations
from loss import nt_xent_loss
from model import PalmprintEncoder

PREPROCESSED_DATA_DIR = '../dataset/preprocessed_images'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
EMBEDDING_DIM = 138
TEMPERATURE = 0.07
IMAGE_SIZE = 138


class PalmprintDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        image = np.expand_dims(image, axis=0)  # Add channel dimension (grayscale -> 1 channel)
        image = torch.from_numpy(image).float()  # Convert to torch tensor, float type

        if self.transform:
            augmented_image_1 = self.transform(
                image.repeat(3, 1, 1))  # Apply augmentations to get view 1 (repeat to 3 channel for transforms)
            augmented_image_2 = self.transform(image.repeat(3, 1, 1))  # Apply augmentations again for view 2

            return augmented_image_1, augmented_image_2

        return image, image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PalmprintDataset(PREPROCESSED_DATA_DIR, transform=get_palmprint_augmentations(IMAGE_SIZE))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                        pin_memory=True)  # DataLoader for efficient batching

model = PalmprintEncoder(embedding_dim=EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (img1_batch, img2_batch) in enumerate(dataloader):
        img1_batch, img2_batch = img1_batch.to(device), img2_batch.to(device)

        optimizer.zero_grad()

        # Get embeddings for both augmented views
        embeddings_1 = model(img1_batch)  # Shape: [batch_size, embedding_dim]
        embeddings_2 = model(img2_batch)  # Shape: [batch_size, embedding_dim]

        # Concatenate embeddings to create input for NT-Xent loss
        embeddings = torch.cat([embeddings_1, embeddings_2], dim=0)  # Shape: [batch_size * 2, embedding_dim]

        loss = nt_xent_loss(embeddings, temperature=TEMPERATURE)  # Calculate NT-Xent loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

        if batch_idx % 10 == 0:  # Print loss every 10 batches
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Avg Loss: {avg_loss:.4f}, Batch Loss: {loss.item():.4f}")

    avg_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] - Average Loss: {avg_epoch_loss:.4f}")

# --- Save Trained Encoder ---
encoder_save_path = '../output/model/palmprint_encoder.pth'  # Path to save trained encoder weights
torch.save(model.encoder.state_dict(), encoder_save_path)  # Save only the encoder part of the model
print(f"Trained Palmprint Encoder saved to: {encoder_save_path}")
