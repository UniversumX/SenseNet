import numpy as np
import os

def synchronize_data(eeg_data_path, proprio_data_path, visual_data_path, save_path):
    """
    Synchronize EEG, proprioceptive, and visual data based on shared time vectors.
    """
    eeg_data = np.load(eeg_data_path)
    proprio_data = np.load(proprio_data_path)
    visual_data = np.load(visual_data_path)
    # Assuming data are already aligned, but you can implement alignment logic here
    # For now, we will just save the datasets together
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'eeg_data_sync.npy'), eeg_data)
    np.save(os.path.join(save_path, 'proprio_data_sync.npy'), proprio_data)
    np.save(os.path.join(save_path, 'visual_data_sync.npy'), visual_data)
    print(f"Synchronized data saved to {save_path}")

if __name__ == '__main__':
    synchronize_data(
        eeg_data_path='data/processed/eeg_data_preprocessed.npy',
        proprio_data_path='data/processed/proprio_data_preprocessed.npy',
        visual_data_path='data/processed/visual_data_preprocessed.npy',
        save_path='data/processed'
    )