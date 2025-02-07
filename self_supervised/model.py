import torch.nn as nn
import torchvision.models as models


class PalmprintEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        # Use a pre-trained ResNet-18 (remove classification head)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer (avgpool + fc)

        # Projection Head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, resnet.fc.in_features),  # First Linear layer
            nn.ReLU(),
            nn.Linear(resnet.fc.in_features, embedding_dim)  # Second Linear layer to embedding_dim
        )

    def forward(self, x):
        features = self.encoder(x)  # Output shape: [batch_size, 512, 1, 1]
        features = features.flatten(1)  # Flatten to: [batch_size, 512]
        embeddings = self.projection_head(features)  # Output shape: [batch_size, embedding_dim]
        return embeddings
