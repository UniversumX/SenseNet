import numpy as np
import mne
from mne.simulation import simulate_raw
import os
from config import cfg

def generate_synthetic_eeg(save_path):
    """
    Generate realistic synthetic EEG data using MNE's simulation capabilities.
    """
    os.makedirs(save_path, exist_ok=True)
    info = mne.create_info(
        ch_names=['EEG %03d' % i for i in range(cfg.data.eeg_channels)],
        sfreq=cfg.data.sampling_rate,
        ch_types='eeg'
    )
    times = np.arange(0, cfg.data.sequence_length / cfg.data.sampling_rate, 1 / cfg.data.sampling_rate)
    n_samples = cfg.data.n_samples
    eeg_data = []

    for _ in range(n_samples):
        # Simulate EEG data with multiple sine waves and noise
        signals = []
        for _ in range(cfg.data.eeg_channels):
            freq = np.random.uniform(1, 40)  # EEG frequencies
            amplitude = np.random.uniform(5, 50)  # EEG amplitudes
            phase = np.random.uniform(0, 2 * np.pi)
            signal = amplitude * np.sin(2 * np.pi * freq * times + phase)
            signals.append(signal)
        signals = np.array(signals)
        # Add Gaussian noise
        noise = np.random.normal(0, 1, signals.shape)
        signals += noise
        eeg_data.append(signals)

    eeg_data = np.stack(eeg_data)
    np.save(os.path.join(save_path, 'eeg_data.npy'), eeg_data)
    print(f"Synthetic EEG data saved to {save_path}")

if __name__ == '__main__':
    generate_synthetic_eeg(save_path='data/synthetic')