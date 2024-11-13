"""
import torch
import torch.nn as nn

class EEGEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(EEGEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, latent_dim, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x, _ = self.lstm(x)
        return x[:, -1, :]  # Return the last time step

# Example usage:
# encoder = EEGEncoder(input_size=64, hidden_size=128, latent_dim=256)
"""

import torch.nn as nn
from models.transformer_encoder import TransformerEEGEncoder
# from models.cnn_lstm_encoder import CNNLSTMEncoder  # implement this

def get_eeg_encoder(cfg):
    if cfg.model.encoder == 'transformer':
        encoder = TransformerEEGEncoder(
            input_dim=cfg.data.eeg_channels,
            num_heads=8,
            num_layers=4,
            hidden_dim=128,
            latent_dim=cfg.model.latent_dim
        )
    elif cfg.model.encoder == 'cnn_lstm':
        encoder = CNNLSTMEncoder(
            input_size=cfg.data.eeg_channels,
            hidden_size=128,
            latent_dim=cfg.model.latent_dim
        )
    else:
        raise ValueError("Invalid encoder type specified.")
    return encoder