import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

class EEGPilotDataset(Dataset):
    def __init__(self, config):
        """
        Dataset for EEG + auxiliary data (gyroscope, visual features)
        
        Data organization:
        - EEG: (batch, channels, time)
        - Gyro angles: (batch, 3) - binned into ranges
        - Visual features: (batch, n_features) - binned into ranges
        """
        self.config = config
        self.eeg_data = []
        self.gyro_data = []
        self.visual_data = []
        
        # Discretization bins for curriculum learning
        self.angle_bins = np.linspace(-180, 180, config.n_angle_bins + 1)
        self.feature_bins = np.linspace(0, 1, config.n_feature_bins + 1)
        
    def load_data(self, eeg_path: str, gyro_path: str, visual_path: str):
        """Load and preprocess data"""
        # Load EEG data (channels x time)
        self.eeg_data = torch.load(eeg_path)
        
        # Load and bin gyroscope angles
        gyro_data = torch.load(gyro_path)
        self.gyro_data = np.digitize(gyro_data, self.angle_bins) - 1
        
        # Load and bin visual features
        visual_data = torch.load(visual_path)
        self.visual_data = np.digitize(visual_data, self.feature_bins) - 1
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_data[idx],
            'gyro': self.gyro_data[idx],
            'visual': self.visual_data[idx]
        }

class EEGPilotHydraNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # EEG Encoder (Shared Stem)
        self.eeg_encoder = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 16, kernel_size=(1, config.kernel_size)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Spatial convolution
            nn.Conv2d(16, 32, kernel_size=(config.n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Feature extraction
            nn.Conv2d(32, 64, kernel_size=(1, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 10)),
            nn.Flatten()
        )
        
        # Task-specific heads
        self.proprioception_head = self._create_head(
            output_size=config.n_angle_bins
        )
        
        self.visual_head = self._create_head(
            output_size=config.n_feature_bins
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
    def _create_head(self, output_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode EEG
        features = self.eeg_encoder(x.unsqueeze(1))
        
        # Get gating weights
        gates = self.gate(features)
        
        outputs = {}
        if gates[0, 0] > 0.5:  # Proprioception gate
            outputs['proprioception'] = self.proprioception_head(features)
        if gates[0, 1] > 0.5:  # Visual gate
            outputs['visual'] = self.visual_head(features)
            
        return outputs, gates

class CurriculumTrainer:
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Task-specific losses
        self.proprioception_loss = nn.CrossEntropyLoss()
        self.visual_loss = nn.CrossEntropyLoss()
        
    def train_epoch(self, 
                   dataloader: DataLoader, 
                   curriculum_stage: int):
        """
        Train for one epoch with curriculum learning
        
        curriculum_stage:
            1: Coarse bins (early training)
            2: Medium bins
            3: Fine bins
        """
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, gates = self.model(batch['eeg'])
            
            # Calculate losses based on curriculum stage
            loss = 0
            if 'proprioception' in outputs:
                prop_loss = self.proprioception_loss(
                    outputs['proprioception'], 
                    self._adjust_labels(batch['gyro'], curriculum_stage)
                )
                loss += self.config.prop_weight * prop_loss
                
            if 'visual' in outputs:
                vis_loss = self.visual_loss(
                    outputs['visual'],
                    self._adjust_labels(batch['visual'], curriculum_stage)
                )
                loss += self.config.visual_weight * vis_loss
            
            # Add gating regularization
            gate_loss = self.config.gate_weight * (gates.mean() - 0.5).abs()
            loss += gate_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def _adjust_labels(self, 
                      labels: torch.Tensor, 
                      curriculum_stage: int) -> torch.Tensor:
        """Adjust label granularity based on curriculum stage"""
        if curriculum_stage == 1:
            # Coarse bins (reduce to 3 bins)
            return labels // (self.config.n_bins // 3)
        elif curriculum_stage == 2:
            # Medium bins (reduce to 5 bins)
            return labels // (self.config.n_bins // 5)
        else:
            # Fine bins (use all bins)
            return labels

class EEGPilotConfig:
    def __init__(self):
        # Data configuration
        self.n_channels = 64
        self.n_angle_bins = 15  # Initial number of bins for joint angles
        self.n_feature_bins = 10  # Initial number of bins for visual features
        self.kernel_size = 32
        
        # Training configuration
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.prop_weight = 1.0
        self.visual_weight = 1.0
        self.gate_weight = 0.1
        
        # Curriculum stages
        self.curriculum_epochs = {
            1: 10,  # Coarse bins
            2: 20,  # Medium bins
            3: 30   # Fine bins
        }

def run_pilot():
    # Initialize configuration
    config = EEGPilotConfig()
    
    # Create dataset
    dataset = EEGPilotDataset(config)
    dataset.load_data(
        eeg_path='path/to/eeg.pt',
        gyro_path='path/to/gyro.pt',
        visual_path='path/to/visual.pt'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Initialize model
    model = EEGPilotHydraNet(config).cuda()
    
    # Initialize trainer
    trainer = CurriculumTrainer(model, config)
    
    # Training loop with curriculum
    for stage in [1, 2, 3]:
        print(f"Starting curriculum stage {stage}")
        for epoch in range(config.curriculum_epochs[stage]):
            loss = trainer.train_epoch(dataloader, stage)
            print(f"Stage {stage}, Epoch {epoch+1}, Loss: {loss:.4f}")
