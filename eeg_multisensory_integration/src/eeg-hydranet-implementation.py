import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

class EEGDataset(Dataset):
    def __init__(self, config, mode='train'):
        """
        Dataset class for EEG data with proprioception and visual fixation tasks
        
        Args:
            config: Configuration object
            mode: 'train', 'val', or 'test'
        """
        self.config = config
        self.mode = mode
        self.eeg_scaler = StandardScaler()
        self.gyro_scaler = StandardScaler()
        self.position_scaler = StandardScaler()
        
        # Load and preprocess data
        self.eeg_data = []  # Shape: (n_samples, n_channels, n_timepoints)
        self.gyro_data = []  # Shape: (n_samples, n_joints, 3)  # xyz angles
        self.position_data = []  # Shape: (n_samples, n_joints, 3)  # xyz positions
        self.visual_features = []  # Shape: (n_samples, n_features)
        
        self._load_data()
        self._preprocess_data()

    def _load_data(self):
        """Load raw data files"""
        # Example loading code - adapt to your data format
        raw = mne.io.read_raw_edf(self.config.eeg_file, preload=True)
        events = mne.find_events(raw)
        
        # Extract epochs
        epochs = mne.Epochs(raw, events, event_id=self.config.event_ids, 
                          tmin=self.config.tmin, tmax=self.config.tmax,
                          baseline=None, preload=True)
        
        self.eeg_data = epochs.get_data()
        
        # Load corresponding proprioception and visual data
        # TODO: Implement actual data loading based on your file format
        pass

    def _preprocess_data(self):
        """Apply preprocessing to raw data"""
        # EEG preprocessing
        self.eeg_data = self._bandpass_filter(self.eeg_data)
        self.eeg_data = self._apply_spatial_filter(self.eeg_data)
        
        # Scale data
        self.eeg_data = self.eeg_scaler.fit_transform(self.eeg_data.reshape(-1, self.eeg_data.shape[-1])).reshape(self.eeg_data.shape)
        self.gyro_data = self.gyro_scaler.fit_transform(self.gyro_data.reshape(-1, 3)).reshape(self.gyro_data.shape)
        self.position_data = self.position_scaler.fit_transform(self.position_data.reshape(-1, 3)).reshape(self.position_data.shape)

    def _bandpass_filter(self, data, lowcut=1, highcut=50, fs=1000):
        """Apply bandpass filter to EEG data"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def _apply_spatial_filter(self, data):
        """Apply spatial filtering (e.g., CAR, Laplacian)"""
        # Common Average Reference
        return data - np.mean(data, axis=1, keepdims=True)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        sample = {
            'eeg': torch.FloatTensor(self.eeg_data[idx]),
            'gyro': torch.FloatTensor(self.gyro_data[idx]),
            'position': torch.FloatTensor(self.position_data[idx]),
            'visual_features': torch.FloatTensor(self.visual_features[idx])
        }
        return sample

class EEGHydraNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Initialize weights
        self.sample_wts = [
            torch.distributions.Dirichlet(torch.ones(config.batch_size)).sample().cuda()
            for _ in range(config.n_heads)
        ]
        
        # EEG feature extraction
        self.eeg_encoder = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 32, kernel_size=(1, config.kernel_size), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Spatial convolution
            nn.Conv2d(32, 64, kernel_size=(config.n_channels, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Feature processing
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 10)),
            nn.Flatten()
        )
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.n_joints * 6, 256),  # 6 = 3 (angles) + 3 (positions)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
        # Visual feature encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.n_visual_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
        # Modality fusion
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512
            ),
            num_layers=2
        )
        
        # Task-specific heads
        self.heads = nn.ModuleList([
            TaskHead(config) for _ in range(config.n_heads)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(384, 128),  # 384 = 128 * 3 (fused features)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.n_heads),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from each modality
        eeg_features = self.eeg_encoder(x['eeg'].unsqueeze(1))
        
        # Combine gyro and position data for proprioception
        proprio_input = torch.cat([x['gyro'], x['position']], dim=-1)
        proprio_features = self.proprio_encoder(proprio_input)
        
        visual_features = self.visual_encoder(x['visual_features'])
        
        # Fuse modalities
        fused_features = torch.stack([eeg_features, proprio_features, visual_features], dim=0)
        fused_features = self.fusion(fused_features)
        
        # Get gating weights
        gate_scores = self.gate(torch.cat(list(fused_features), dim=-1))
        
        # Get top-k active heads
        k = self.config.active_heads
        _, top_indices = torch.topk(gate_scores, k)
        
        # Process through selected heads
        outputs = []
        for idx in top_indices[0]:
            head_output = self.heads[idx](fused_features)
            outputs.append(head_output)
            
        return outputs, gate_scores

class TaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8
        )
        
        self.temporal_processor = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.output_dim)
        )
    
    def forward(self, x):
        # Self-attention
        attended, _ = self.attention(x, x, x)
        
        # Temporal processing
        temporal_out, _ = self.temporal_processor(attended)
        
        # Generate output
        return self.output(temporal_out)

# Training configuration
class EEGConfig:
    def __init__(self):
        # Data
        self.eeg_file = 'path/to/eeg/data'
        self.event_ids = {'visual_task': 1, 'proprio_task': 2}
        self.tmin = -0.2
        self.tmax = 1.0
        
        # EEG parameters
        self.n_channels = 64
        self.sampling_rate = 1000
        self.kernel_size = 20
        
        # Architecture
        self.n_heads = 2  # One for each task
        self.active_heads = 1  # Only one task at a time
        self.n_joints = 10
        self.n_visual_features = 50
        self.output_dim = 3  # Depends on task
        
        # Training
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # Loss weights
        self.task_weights = {
            'proprioception': 1.0,
            'visual': 1.0
        }
        self.gate_sparsity_weight = 0.1

def train_epoch(model, dataloader, optimizer, criterion, config):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs, gates = model(batch)
        
        # Calculate loss
        loss = 0
        for task_idx, output in enumerate(outputs):
            task_loss = criterion(output, batch[f'task_{task_idx}_target'])
            loss += config.task_weights[f'task_{task_idx}'] * task_loss
            
        # Add gating regularization
        gate_loss = config.gate_sparsity_weight * (gates.mean() - config.target_sparsity).abs()
        loss += gate_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
