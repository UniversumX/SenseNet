"""
import torch
import torch.nn as nn

class ProprioceptiveDecoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(ProprioceptiveDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example usage:
# decoder = ProprioceptiveDecoder(latent_dim=256, output_size=10)
"""
import torch
import torch.nn as nn

class ProprioceptiveDecoder(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(ProprioceptiveDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.decoder(x)