"""
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_space(encoder, dataloader):
    latent_vectors = []
    labels = []
    for eeg_data, label in dataloader:
        with torch.no_grad():
            latent = encoder(eeg_data)
        latent_vectors.append(latent.numpy())
        labels.append(label.numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    tsne = TSNE(n_components=2)
    latent_2d = tsne.fit_transform(latent_vectors)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels)
    plt.colorbar()
    plt.title('Latent Space Visualization')
    plt.show()
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from captum.attr import Saliency
from utils.data_loader import get_dataloader
from config import cfg
from models.eeg_encoder import get_eeg_encoder

def visualize_latent_space(encoder, dataloader):
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for eeg_data, label in dataloader:
            latent = encoder(eeg_data)
            latent_vectors.append(latent.cpu().numpy())
            labels.append(label.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    latent_2d = tsne.fit_transform(latent_vectors)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.title('Latent Space Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def compute_saliency_maps(encoder, decoder, dataloader):
    saliency = Saliency(decoder)
    for eeg_data, _ in dataloader:
        eeg_data.requires_grad_()
        attribution = saliency.attribute(eeg_data)
        attribution = attribution.cpu().numpy()
        # Plot or save the saliency maps
        break  # For demonstration, we only process one batch

if __name__ == '__main__':
    encoder = get_eeg_encoder(cfg)
    encoder.load_state_dict(torch.load(os.path.join(cfg.paths.model_dir, 'encoder_transfer.pth')))
    encoder.eval()
    dataloader = get_dataloader('visual', train=False)
    visualize_latent_space(encoder, dataloader)