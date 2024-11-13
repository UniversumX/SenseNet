import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from config import cfg

def generate_synthetic_visual(save_path):
    """
    Generate synthetic visual feature data representing object attributes.
    """
    os.makedirs(save_path, exist_ok=True)
    n_samples = cfg.data.n_samples
    visual_data = []

    # Define possible features
    edges_options = [0, 1, 2, 3, 4]  # Number of edges
    shapes_options = ['cube', 'sphere', 'pyramid', 'cylinder', 'torus']
    colors_options = ['red', 'green', 'blue', 'yellow', 'purple']
    sizes_options = ['small', 'medium', 'large']
    textures_options = ['smooth', 'rough', 'striped', 'dotted']

    encoder = OneHotEncoder()
    encoder.fit([edges_options, shapes_options, colors_options, sizes_options, textures_options])

    for _ in range(n_samples):
        features = []
        # Randomly select features
        edges = np.random.choice(edges_options)
        shape = np.random.choice(shapes_options)
        color = np.random.choice(colors_options)
        size = np.random.choice(sizes_options)
        texture = np.random.choice(textures_options)
        # Encode features
        feature_vector = encoder.transform([[edges, shape, color, size, texture]]).toarray()
        features.append(feature_vector.flatten())
        visual_data.append(features)

    visual_data = np.vstack(visual_data)
    np.save(os.path.join(save_path, 'visual_data.npy'), visual_data)
    print(f"Synthetic visual data saved to {save_path}")

if __name__ == '__main__':
    generate_synthetic_visual(save_path='data/synthetic')