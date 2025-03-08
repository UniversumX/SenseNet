import numpy as np
from scipy.signal import butter, filtfilt, chirp
import torch
from sklearn.preprocessing import StandardScaler
import mne


class IntegratedWorldSimulator:
    """
    Simulates an integrated perceptual world with temporal dependencies
    across different modalities and tasks
    """
    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Create base "world states" that influence all modalities
        self.world_states = self._generate_world_states()
        
    def _generate_world_states(self):
        """
        Generate underlying world states that influence all perceptual modalities.
        Uses state space model to create temporally coherent latent states.
        """
        n_states = self.config.n_latent_states
        n_times = self.config.n_timepoints * self.config.n_trials
        
        # Generate smooth temporal transitions in latent space
        states = generate_random_state_space(n_times, n_states, self.rng)[0]
        
        # Reshape to trials x timepoints x states
        states = states.reshape(self.config.n_trials, self.config.n_timepoints, n_states)
        
        return states

    def generate_random_state_space(n_times, n_states, rng):
        # Generate smooth latent state trajectories via cumulative sums
        states = np.cumsum(rng.randn(n_times, n_states), axis=0)
        return states, None  # Adjust return values as needed
    
    def generate_eeg_data(self):
        """
        Generate synthetic EEG data influenced by world states
        """
        n_channels = self.config.n_channels
        n_trials = self.config.n_trials
        n_times = self.config.n_timepoints
        
        # Base oscillations at different frequencies
        times = np.linspace(0, 1, n_times)
        freqs = np.linspace(1, 50, n_channels)  # Different frequency per channel
        
        eeg_data = np.zeros((n_trials, n_channels, n_times))
        
        for trial in range(n_trials):
            # Get world state for this trial
            state = self.world_states[trial]
            
            for ch in range(n_channels):
                # Base oscillation
                signal = chirp(times, freqs[ch], 1, freqs[ch] * 1.5)
                
                # Modulate amplitude based on world states
                state_influence = np.dot(state, self.rng.randn(self.config.n_latent_states))
                modulated_signal = signal * (1 + 0.5 * state_influence)
                
                # Add spatial correlation between channels
                spatial_mixing = self.rng.randn(n_channels) * 0.3
                eeg_data[trial, ch] = modulated_signal + np.dot(spatial_mixing, eeg_data[trial])
        
        # Add noise
        eeg_data += 0.1 * self.rng.randn(*eeg_data.shape)
        
        return eeg_data
    
    def generate_proprioception_data(self):
        """
        Generate synthetic proprioception data (joint angles and positions)
        that correlates with world states and EEG
        """
        n_joints = self.config.n_joints
        n_trials = self.config.n_trials
        n_times = self.config.n_timepoints
        
        # Generate smooth joint trajectories influenced by world states
        angles = np.zeros((n_trials, n_times, n_joints, 3))  # xyz angles
        positions = np.zeros((n_trials, n_times, n_joints, 3))  # xyz positions
        
        for trial in range(n_trials):
            state = self.world_states[trial]
            
            # Base movement patterns
            t = np.linspace(0, 2*np.pi, n_times)
            
            for joint in range(n_joints):
                # Generate smooth angle trajectories
                for axis in range(3):
                    # Base oscillation
                    base_angle = np.sin(t + joint * 0.5)
                    
                    # Modulate based on world state
                    state_influence = np.dot(state, self.rng.randn(self.config.n_latent_states))
                    angles[trial, :, joint, axis] = base_angle * (1 + 0.3 * state_influence)
                
                # Convert angles to positions (simplified forward kinematics)
                positions[trial, :, joint] = self._angles_to_positions(angles[trial, :, joint])
        
        # Add realistic noise
        angles += 0.05 * self.rng.randn(*angles.shape)
        positions += 0.02 * self.rng.randn(*positions.shape)
        
        return angles, positions
    
    def generate_visual_features(self):
        """
        Generate synthetic visual features that correlate with proprioception
        and world states
        """
        n_features = self.config.n_visual_features
        n_trials = self.config.n_trials
        n_times = self.config.n_timepoints
        
        features = np.zeros((n_trials, n_times, n_features))
        
        for trial in range(n_trials):
            state = self.world_states[trial]
            
            # Base features
            base_features = self.rng.randn(n_features)
            
            # Modulate features based on world state
            for t in range(n_times):
                state_influence = np.dot(state[t], self.rng.randn(self.config.n_latent_states, n_features))
                features[trial, t] = base_features + state_influence
        
        # Add temporal smoothing
        features = self._smooth_temporal(features)
        
        return features
    
    def _angles_to_positions(self, angles):
        """
        Simplified forward kinematics to convert angles to positions
        """
        # Simplified - just using angles as offset for position
        positions = np.cumsum(angles, axis=0) * 0.1
        return positions
    
    def _smooth_temporal(self, data, window=5):
        """Apply temporal smoothing"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=window, axis=1)
    
    def generate_integrated_dataset(self):
        """
        Generate complete integrated dataset with all modalities
        """
        print("Generating EEG data...")
        eeg_data = self.generate_eeg_data()
        
        print("Generating proprioception data...")
        angles, positions = self.generate_proprioception_data()
        
        print("Generating visual features...")
        visual_features = self.generate_visual_features()
        
        # Create task labels/targets
        proprio_targets = self._generate_proprio_targets(angles, positions)
        visual_targets = self._generate_visual_targets(visual_features)
        
        dataset = {
            'eeg': eeg_data,
            'angles': angles,
            'positions': positions,
            'visual_features': visual_features,
            'proprio_targets': proprio_targets,
            'visual_targets': visual_targets,
            'world_states': self.world_states  # Include for validation
        }
        
        return dataset
    
    def _generate_proprio_targets(self, angles, positions):
        """Generate proprioception task targets"""
        # Example: predict next timestep positions
        return positions[:, 1:] - positions[:, :-1]
    
    def _generate_visual_targets(self, features):
        """Generate visual task targets"""
        # Example: predict feature changes
        return features[:, 1:] - features[:, :-1]

class DataConfig:
    def __init__(self):
        # Data dimensions
        self.n_trials = 1000
        self.n_timepoints = 1000
        self.n_channels = 64
        self.n_joints = 10
        self.n_visual_features = 50
        self.n_latent_states = 20
        
        # Simulation parameters
        self.seed = 42
        self.sampling_rate = 1000  # Hz
        
        # Task parameters
        self.n_classes = 5  # For classification tasks
        self.noise_level = 0.1

def validate_dependencies(dataset):
    """
    Validate that generated data has meaningful statistical dependencies
    """
    from scipy.stats import pearsonr
    
    # Check temporal correlations
    temporal_corr_eeg = np.mean([pearsonr(dataset['eeg'][i, 0, :-1], 
                                         dataset['eeg'][i, 0, 1:])[0] 
                                for i in range(10)])
    
    # Check cross-modality correlations
    eeg_visual_corr = np.mean([pearsonr(np.mean(dataset['eeg'][i], axis=0),
                                       np.mean(dataset['visual_features'][i], axis=1))[0]
                              for i in range(10)])
    
    print(f"Temporal correlation in EEG: {temporal_corr_eeg:.3f}")
    print(f"EEG-Visual correlation: {eeg_visual_corr:.3f}")

if __name__ == "__main__":
    # Generate synthetic dataset
    config = DataConfig()
    simulator = IntegratedWorldSimulator(config)
    dataset = simulator.generate_integrated_dataset()
    
    # Validate dependencies
    validate_dependencies(dataset)
    
    # Save dataset
    np.save('synthetic_integrated_data.npy', dataset)
    
    print("Dataset generated and saved!")
    print(f"EEG shape: {dataset['eeg'].shape}")
    print(f"Angles shape: {dataset['angles'].shape}")
    print(f"Positions shape: {dataset['positions'].shape}")
    print(f"Visual features shape: {dataset['visual_features'].shape}")
