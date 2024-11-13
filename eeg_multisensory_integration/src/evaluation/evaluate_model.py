"""
import torch
from models.encoder import EEGEncoder
from models.visual_decoder import VisualDecoder
from utils.data_loader import get_dataloader
from sklearn.metrics import mean_squared_error

def evaluate_visual_model():
    # Load trained encoder
    encoder = EEGEncoder(input_size=64, hidden_size=128, latent_dim=256)
    encoder.load_state_dict(torch.load('models/encoder_transfer.pth'))
    # Initialize visual decoder
    decoder = VisualDecoder(latent_dim=256, output_size=20)
    decoder.load_state_dict(torch.load('models/visual_decoder.pth'))
    # Load data
    dataloader = get_dataloader('visual', train=False)
    # Evaluation loop
    all_preds = []
    all_labels = []
    for eeg_data, visual_data in dataloader:
        with torch.no_grad():
            latent = encoder(eeg_data)
            output = decoder(latent)
        all_preds.append(output.numpy())
        all_labels.append(visual_data.numpy())
    mse = mean_squared_error(all_labels, all_preds)
    print(f"Mean Squared Error on test set: {mse}")

if __name__ == '__main__':
    evaluate_visual_model()
"""

import torch
import os
import numpy as np
from models.eeg_encoder import get_eeg_encoder
from models.visual_decoder import VisualDecoder
from utils.data_loader import get_dataloader
from utils.metrics import compute_regression_metrics
from config import cfg

def evaluate_visual_model():
    # Load trained encoder
    encoder = get_eeg_encoder(cfg)
    encoder.load_state_dict(torch.load(os.path.join(cfg.paths.model_dir, 'encoder_transfer.pth')))
    encoder.eval()
    # Initialize visual decoder
    decoder = VisualDecoder(latent_dim=cfg.model.latent_dim, output_size=cfg.data.visual_feature_dim)
    decoder.load_state_dict(torch.load(os.path.join(cfg.paths.model_dir, 'visual_decoder.pth')))
    decoder.eval()
    # Load data
    dataloader = get_dataloader('visual', train=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for eeg_data, visual_data in dataloader:
            latent = encoder(eeg_data)
            output = decoder(latent)
            all_preds.append(output.cpu().numpy())
            all_labels.append(visual_data.cpu().numpy())
    # Compute metrics
    metrics = compute_regression_metrics(np.vstack(all_labels), np.vstack(all_preds))
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    evaluate_visual_model()