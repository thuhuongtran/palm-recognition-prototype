import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(AutoEncoder, self).__init__()
        # Encoder: simple CNN
        self.enc_conv1 = nn.Conv2d(1,
                                   16, 3, stride=2, padding=1)  # grayscale => 1 channel
        self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        # flatten + linear
        self.fc_enc = nn.Linear(64 * (224//8) * (224//8), latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64 * (224//8) * (224//8))

        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec_deconv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_enc(x)         # (batch_size, latent_dim)
        return x

    def decoder(self, z):
        x = self.fc_dec(z)  # (batch_size, 64*28*28) if 224/8=28
        x = x.view(x.size(0), 64, (224//8), (224//8))
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = torch.sigmoid(self.dec_deconv3(x))  # output in [0, 1]
        return x

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
