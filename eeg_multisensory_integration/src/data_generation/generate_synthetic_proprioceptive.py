import numpy as np
import os
from config import cfg

def generate_synthetic_proprioceptive(save_path):
    """
    Generate synthetic proprioceptive data representing realistic arm movements.
    """
    os.makedirs(save_path, exist_ok=True)
    n_samples = cfg.data.n_samples
    n_joints = 7  # Number of joints in the arm (shoulder, elbow, wrist)
    sequence_length = cfg.data.sequence_length
    sampling_rate = cfg.data.sampling_rate
    times = np.linspace(0, sequence_length / sampling_rate, sequence_length)
    proprio_data = []

    for _ in range(n_samples):
        joint_angles = []
        for _ in range(n_joints):
            freq = np.random.uniform(0.1, 1.0)  # Slow movements
            amplitude = np.random.uniform(0, 90)  # Joint angles in degrees
            phase = np.random.uniform(0, 2 * np.pi)
            angle = amplitude * np.sin(2 * np.pi * freq * times + phase)
            joint_angles.append(angle)
        joint_angles = np.array(joint_angles)
        proprio_data.append(joint_angles)

    proprio_data = np.stack(proprio_data)
    np.save(os.path.join(save_path, 'proprio_data.npy'), proprio_data)
    print(f"Synthetic proprioceptive data saved to {save_path}")

if __name__ == '__main__':
    generate_synthetic_proprioceptive(save_path='data/synthetic')