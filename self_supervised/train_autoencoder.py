import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PalmDataset, get_transforms
from autoencoder import AutoEncoder


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, _ in dataloader:
        images = images.to(device)  # shape: (B, 1, 224, 224) if grayscale

        # Forward pass
        recon = model(images)
        loss = criterion(recon, images)  # e.g., MSELoss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss


def main():
    # Paths
    train_dir = "../dataset/split_data/train"

    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = 5
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    transform = get_transforms()  # from dataset.py
    train_dataset = PalmDataset(root_dir=train_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Model, Loss, Optimizer
    model = AutoEncoder(latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/autoencoder.pth")
    print("Model saved to checkpoints/autoencoder.pth")


if __name__ == "__main__":
    main()
