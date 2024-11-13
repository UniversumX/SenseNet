import numpy as np
import os
from scipy.signal import savgol_filter
from config import cfg

def preprocess_proprioceptive(proprio_data_path, save_path):
    """
    Preprocess proprioceptive data: smoothing, normalization, feature extraction.
    """
    proprio_data = np.load(proprio_data_path)
    n_samples, n_joints, n_times = proprio_data.shape
    processed_data = []

    for i in range(n_samples):
        data = proprio_data[i]
        # Smoothing
        data_smoothed = savgol_filter(data, window_length=11, polyorder=3, axis=1)
        # Normalization
        data_smoothed = (data_smoothed - np.mean(data_smoothed)) / np.std(data_smoothed)
        processed_data.append(data_smoothed)

    processed_data = np.stack(processed_data)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'proprio_data_preprocessed.npy'), processed_data)
    print(f"Preprocessed proprioceptive data saved to {save_path}")

if __name__ == '__main__':
    preprocess_proprioceptive(
        proprio_data_path='data/synthetic/proprio_data.npy',
        save_path='data/processed'
    )