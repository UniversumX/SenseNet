"""
import torch

def curriculum_learning_scheduler(epoch, total_epochs):

    # Adjust learning parameters or data complexity over epochs.

    # Example: Linearly increase complexity
    complexity = epoch / total_epochs
    return complexity

"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_curriculum_dataloader(dataset, stage, total_stages, batch_size):
    """
    Adjust the dataset complexity based on the curriculum stage.
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    # Linearly increase the amount of data
    current_sample_size = int(n_samples * (stage + 1) / total_stages)
    current_indices = indices[:current_sample_size]
    current_dataset = Subset(dataset, current_indices)
    dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)
    return dataloader