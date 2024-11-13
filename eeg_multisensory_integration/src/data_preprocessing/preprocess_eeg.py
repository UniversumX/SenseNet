import numpy as np
import mne
import os
from mne.preprocessing import ICA, create_eog_epochs
from config import cfg

def preprocess_eeg(eeg_data_path, save_path):
    """
    Preprocess EEG data: filtering, artifact removal using ICA, normalization.
    """
    eeg_data = np.load(eeg_data_path)
    n_samples, n_channels, n_times = eeg_data.shape
    sfreq = cfg.data.sampling_rate
    processed_data = []

    for i in range(n_samples):
        raw = mne.io.RawArray(eeg_data[i], mne.create_info(n_channels, sfreq, ch_types='eeg'))
        # Band-pass filter
        raw.filter(1., 40., fir_design='firwin')
        # Set EEG reference
        raw.set_eeg_reference('average', projection=True)
        # Artifact removal using ICA
        ica = ICA(n_components=15, random_state=97)
        ica.fit(raw)
        eog_epochs = create_eog_epochs(raw)
        eog_indices, eog_scores = ica.find_bads_eog(eog_epochs)
        ica.exclude = eog_indices
        raw_corrected = ica.apply(raw)
        # Get data
        data = raw_corrected.get_data()
        # Normalization
        data = (data - np.mean(data)) / np.std(data)
        processed_data.append(data)

    processed_data = np.stack(processed_data)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'eeg_data_preprocessed.npy'), processed_data)
    print(f"Preprocessed EEG data saved to {save_path}")

if __name__ == '__main__':
    preprocess_eeg(
        eeg_data_path='data/synthetic/eeg_data.npy',
        save_path='data/processed'
    )