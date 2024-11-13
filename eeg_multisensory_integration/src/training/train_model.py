"""
import torch
from models.encoder import EEGEncoder
from models.proprioceptive_decoder import ProprioceptiveDecoder
from models.visual_decoder import VisualDecoder
from utils.data_loader import get_dataloader

def train_proprioceptive_model():
    # Initialize models
    encoder = EEGEncoder(input_size=64, hidden_size=128, latent_dim=256)
    decoder = ProprioceptiveDecoder(latent_dim=256, output_size=10)
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    # Load data
    dataloader = get_dataloader('proprioceptive')
    # Training loop
    for epoch in range(100):
        for eeg_data, proprio_data in dataloader:
            optimizer.zero_grad()
            latent = encoder(eeg_data)
            output = decoder(latent)
            loss = criterion(output, proprio_data)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss={loss.item()}")

    # Save the trained encoder
    torch.save(encoder.state_dict(), 'models/encoder_proprioceptive.pth')

if __name__ == '__main__':
    train_proprioceptive_model()
"""

import torch
from models.eeg_encoder import get_eeg_encoder
from models.proprioceptive_decoder import ProprioceptiveDecoder
from models.visual_decoder import VisualDecoder
from utils.data_loader import get_dataloader
from utils.metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter
from config import cfg
import os

def train_proprioceptive_model():
    # Initialize models
    encoder = get_eeg_encoder(cfg)
    decoder = ProprioceptiveDecoder(latent_dim=cfg.model.latent_dim, output_size=cfg.data.n_joints)
    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=cfg.paths.log_dir)
    # Load data
    dataset = get_dataloader('proprioceptive', train=True, return_dataset=True)
    total_stages = cfg.curriculum.stages
    for stage in range(total_stages):
        dataloader = get_curriculum_dataloader(dataset, stage, total_stages, cfg.training.batch_size)
        for epoch in range(cfg.training.epochs // total_stages):
            encoder.train()
            decoder.train()
            epoch_loss = 0
            for eeg_data, proprio_data in dataloader:
                optimizer.zero_grad()
                latent = encoder(eeg_data)
                output = decoder(latent)
                loss = criterion(output, proprio_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader)
            writer.add_scalar('Loss/train_stage_{}'.format(stage), epoch_loss, epoch)
            print(f"Stage {stage}, Epoch {epoch}, Loss: {epoch_loss}")
            scheduler.step(epoch_loss)
    # Save the trained encoder
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(cfg.paths.model_dir, 'encoder_proprioceptive.pth'))
    writer.close()


# Similarly, you would implement the transfer learning phase to train the encoder on visual tasks.


if __name__ == '__main__':
    train_proprioceptive_model()