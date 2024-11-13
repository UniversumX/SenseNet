"""
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, eeg_data_path, target_data_path):
        self.eeg_data = np.load(eeg_data_path)
        self.target_data = np.load(target_data_path)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = self.eeg_data[idx]
        target_sample = self.target_data[idx]
        return torch.tensor(eeg_sample, dtype=torch.float32), torch.tensor(target_sample, dtype=torch.float32)

def get_dataloader(task, batch_size=32, train=True):
    if task == 'proprioceptive':
        eeg_data_path = 'data/processed/eeg_data_sync.npy'
        target_data_path = 'data/processed/proprio_data_sync.npy'
    elif task == 'visual':
        eeg_data_path = 'data/processed/eeg_data_sync.npy'
        target_data_path = 'data/processed/visual_data_sync.npy'
    dataset = EEGDataset(eeg_data_path, target_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import cfg

class EEGDataset(Dataset):
    def __init__(self, eeg_data_path, target_data_path, transform=None):
        self.eeg_data = np.load(eeg_data_path)
        self.target_data = np.load(target_data_path)
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = self.eeg_data[idx]
        target_sample = self.target_data[idx]
        if self.transform:
            eeg_sample = self.transform(eeg_sample)
        return torch.tensor(eeg_sample, dtype=torch.float32), torch.tensor(target_sample, dtype=torch.float32)

def get_dataloader(task, train=True, batch_size=None, return_dataset=False):
    if batch_size is None:
        batch_size = cfg.training.batch_size
    if task == 'proprioceptive':
        eeg_data_path = 'data/processed/eeg_data_sync.npy'
        target_data_path = 'data/processed/proprio_data_sync.npy'
    elif task == 'visual':
        eeg_data_path = 'data/processed/eeg_data_sync.npy'
        target_data_path = 'data/processed/visual_data_sync.npy'
    dataset = EEGDataset(eeg_data_path, target_data_path)
    if return_dataset:
        return dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return dataloader