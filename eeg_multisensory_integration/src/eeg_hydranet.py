import torch
import torch.nn as nn
import numpy as np
import mne
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

class EEGDataset(Dataset):
    def __init__(self, data_dict, config=None, mode='train'):
        """
        Dataset class for EEG data with proprioception and visual fixation tasks
        Args:
            data_dict: Dictionary containing the data
            config: Configuration object
            mode: 'train', 'val', or 'test'
        """
        self.config = config
        self.mode = mode
        
        # Initialize scalers
        self.eeg_scaler = StandardScaler()
        self.gyro_scaler = StandardScaler()
        self.position_scaler = StandardScaler()
        
        # Load data from dictionary
        self.eeg_data = data_dict['eeg']
        self.gyro_data = data_dict['angles']
        self.position_data = data_dict['positions']
        self.visual_features = data_dict['visual_features']
        self.proprio_targets = data_dict['proprio_targets']
        self.visual_targets = data_dict['visual_targets']
        
        self._preprocess_data()

    def _preprocess_data(self):
        """Apply preprocessing to raw data"""
        # EEG preprocessing
        self.eeg_data = self._bandpass_filter(self.eeg_data)
        self.eeg_data = self._apply_spatial_filter(self.eeg_data)
        
        # Scale data
        self.eeg_data = self.eeg_scaler.fit_transform(
            self.eeg_data.reshape(-1, self.eeg_data.shape[-1])
        ).reshape(self.eeg_data.shape)
        
        self.gyro_data = self.gyro_scaler.fit_transform(
            self.gyro_data.reshape(-1, self.gyro_data.shape[-1])
        ).reshape(self.gyro_data.shape)
        
        self.position_data = self.position_scaler.fit_transform(
            self.position_data.reshape(-1, self.position_data.shape[-1])
        ).reshape(self.position_data.shape)

    def _bandpass_filter(self, data, lowcut=1, highcut=50, fs=1000):
        """Apply bandpass filter to EEG data"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def _apply_spatial_filter(self, data):
        """Apply spatial filtering (CAR)"""
        return data - np.mean(data, axis=1, keepdims=True)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        sample = {
            'eeg': torch.FloatTensor(self.eeg_data[idx]),
            'gyro': torch.FloatTensor(self.gyro_data[idx]),
            'position': torch.FloatTensor(self.position_data[idx]),
            'visual_features': torch.FloatTensor(self.visual_features[idx]),
            'proprio_targets': torch.FloatTensor(self.proprio_targets[idx]),
            'visual_targets': torch.FloatTensor(self.visual_targets[idx])
        }
        return sample

class EEGHydraNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize weights
        self.sample_wts = [
            torch.distributions.Dirichlet(torch.ones(config.batch_size)).sample()
            for _ in range(config.n_heads)
        ]

        # EEG feature extraction
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(config.n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.n_joints * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Visual feature encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.n_visual_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Modality fusion
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=384,  # Adjusted to match concatenated feature size
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=2
        )

        # Feature reduction layer
        self.feature_reduction = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.n_heads),
            nn.Sigmoid()
        )

        # Task-specific heads
        # Create heads for proprioception and visual tasks
        self.heads = nn.ModuleList([
            TaskHead(config, 'proprio'),  # Head for proprioception task
            TaskHead(config, 'visual')    # Head for visual task
        ])

    def forward(self, x):
        batch_size = x['eeg'].size(0)
        seq_len = x['eeg'].size(2)

        # EEG Input: [batch_size, n_channels, seq_len]
        eeg_input = x['eeg']

        # Process EEG features
        eeg_features = self.eeg_encoder(eeg_input)  # [batch_size, 128, seq_len]
        eeg_features = eeg_features.permute(0, 2, 1)  # [batch_size, seq_len, 128]

        # Proprioception Input: [batch_size, seq_len, n_joints * 6]
        proprio_input = torch.cat([
            x['gyro'].reshape(batch_size, seq_len, -1),
            x['position'].reshape(batch_size, seq_len, -1)
        ], dim=-1)

        proprio_features = self.proprio_encoder(proprio_input)  # [batch_size, seq_len, 128]

        # Visual Features: [batch_size, seq_len, n_visual_features]
        visual_features = self.visual_encoder(x['visual_features'])  # [batch_size, seq_len, 128]

        # Stack features along the feature dimension
        fused_features = torch.cat([
            eeg_features,
            proprio_features,
            visual_features
        ], dim=-1)  # [batch_size, seq_len, 384]

        # Fusion through Transformer
        fused_features = self.fusion(fused_features)  # [batch_size, seq_len, 384]

        # Compute gate scores before feature reduction
        gate_scores = self.gate(fused_features)  # [batch_size, seq_len, n_heads]

        # Reduce feature dimension
        fused_features = self.feature_reduction(fused_features)  # [batch_size, seq_len, 128]

        # Pass through task heads
        outputs = []
        for head in self.heads:
            head_output = head(fused_features)  # [batch_size, seq_len, output_dim]
            outputs.append(head_output)

        return outputs, gate_scores

# Update TaskHead class to handle different output dimensions
class TaskHead(nn.Module):
    def __init__(self, config, task_type):
        super().__init__()
        self.task_type = task_type
        
        # Set output dimension based on task type
        if task_type == 'proprio':
            output_dim = config.proprio_output_dim
        else:  # visual
            output_dim = config.visual_output_dim
        
        self.lstm = nn.LSTM(
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
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.output(lstm_out)

class EEGConfig:
    def __init__(self):
        # Data parameters
        self.n_channels = 64
        self.sampling_rate = 1000
        self.kernel_size = 20
        self.n_timepoints = 1000
        
        # Architecture parameters
        self.n_heads = 2
        self.active_heads = 1
        self.n_joints = 10
        self.n_visual_features = 50
        self.output_dim = self.n_joints * 3  # Adjusted for sequence output

        # Update output dimensions for each task
        self.proprio_output_dim = self.n_joints * 3  # Flattened position deltas
        self.visual_output_dim = self.n_visual_features  # Feature deltas

        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.n_epochs = 100
        self.target_sparsity = 0.5
        
        # Task weights
        self.task_weights = {
            'proprioception': 1.0,
            'visual': 1.0
        }
        self.gate_sparsity_weight = 0.1