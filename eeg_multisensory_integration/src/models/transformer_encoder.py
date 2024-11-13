import torch
import torch.nn as nn
from einops import rearrange

class TransformerEEGEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, latent_dim, dropout=0.1):
        super(TransformerEEGEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, channels, sequence_length]
        """
        x = rearrange(x, 'b c t -> b t c')
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # [sequence_length, batch_size, hidden_dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.dropout(x)
        latent = self.latent_projection(x)
        return latent

# Example usage:
# encoder = TransformerEEGEncoder(input_dim=64, num_heads=8, num_layers=4, hidden_dim=128, latent_dim=256)