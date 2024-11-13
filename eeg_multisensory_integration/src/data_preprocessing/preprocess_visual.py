import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from config import cfg

def preprocess_visual(visual_data_path, save_path):
    """
    Preprocess visual data: normalization.
    """
    visual_data = np.load(visual_data_path)
    scaler = StandardScaler()
    visual_data_scaled = scaler.fit_transform(visual_data)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, 'visual_data_preprocessed.npy'), visual_data_scaled)
    print(f"Preprocessed visual data saved to {save_path}")

if __name__ == '__main__':
    preprocess_visual(
        visual_data_path='data/synthetic/visual_data.npy',
        save_path='data/processed'
    )