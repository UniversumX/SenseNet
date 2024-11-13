# SenseNet: EEG Multisensory Integration Project

## Overview

This project investigates the shared neural representations underlying proprioceptive and visual perception using EEG data. We employ state-of-the-art neural network architectures, curriculum learning, and cross-modal transfer learning to model and analyze multisensory integration.

## Project Structure

eeg_multisensory_integration/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── datasets_info.txt
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── src/
│   ├── config.py
│   ├── data_generation/
│   │   ├── generate_synthetic_eeg.py
│   │   ├── generate_synthetic_proprioceptive.py
│   │   └── generate_synthetic_visual.py
│   ├── data_preprocessing/
│   │   ├── preprocess_eeg.py
│   │   ├── preprocess_proprioceptive.py
│   │   └── preprocess_visual.py
│   ├── data_synchronization/
│   │   └── synchronize_data.py
│   ├── models/
│   │   ├── eeg_encoder.py
│   │   ├── proprioceptive_decoder.py
│   │   ├── visual_decoder.py
│   │   ├── transformer_encoder.py
│   │   └── model_utils.py
│   ├── training/
│   │   ├── curriculum_learning.py
│   │   ├── transfer_learning.py
│   │   └── train_model.py
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   ├── interpretability.py
│   │   └── comparative_analysis.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── logs/
│       └── training_logs/
├── docs/
│   ├── literature_review.md
│   ├── experimental_protocol.md
│   ├── ethics_application.md
│   └── equipment_plan.md
└── scripts/
    ├── run_data_generation.sh
    ├── run_preprocessing.sh
    ├── run_training.sh
    ├── run_evaluation.sh
    └── run_comparative_analysis.sh

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/eeg_multisensory_integration.git
   cd eeg_multisensory_integration
    ```
2.	**Create a virtual environment:**
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.	**Install the required packages:**
pip install -r requirements.txt

How to Run

	1.	Generate synthetic data:
        bash scripts/run_data_generation.sh
    2.	Preprocess the data:
        bash scripts/run_preprocessing.sh
    3.	Train the models:
        bash scripts/run_training.sh
	4.	Evaluate the models:
        bash scripts/run_evaluation.sh
	5.	Run comparative analysis:
        bash scripts/run_comparative_analysis.sh

Documentation

Refer to the docs/ directory for detailed documentation on each deliverable.

Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

License

This project is licensed under the MIT License - see the LICENSE file for details.


### 2. `requirements.txt`

Updated to include the latest versions and additional packages for EEG processing and deep learning.

numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
scipy>=1.7.0
mne>=0.23.0
torch>=1.9.0
torchvision>=0.10.0
tqdm>=4.61.0
seaborn>=0.11.0
tensorboard>=2.5.0
einops>=0.3.0
hydra-core>=1.1.0

### 3. `data/`

No changes needed; the structure remains suitable for data storage.

### 4. `notebooks/`

Enhanced notebooks with thorough exploratory data analysis, model training with visualizations, and comprehensive results analysis using advanced statistical methods.

#### 4.1 `exploratory_data_analysis.ipynb`

Includes:

- Visualization of synthetic EEG signals with spectral analysis.
- Statistical properties of the synthetic data.
- Cross-correlation analysis between modalities.

#### 4.2 `model_training.ipynb`

Includes:

- Interactive training loops with TensorBoard integration.
- Hyperparameter tuning using tools like Optuna or Hyperopt.
- Visualization of training metrics.

#### 4.3 `results_analysis.ipynb`

Includes:

- Detailed performance metrics.
- Comparative plots between different models.
- Statistical significance testing of results.

### 5. `src/`

#### 5.1 `config.py`

A centralized configuration file using Hydra for managing hyperparameters and paths.

```python
# config.py
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'data': {
        'eeg_channels': 64,
        'sampling_rate': 256,
        'sequence_length': 1024,
        'n_samples': 10000,
    },
    'model': {
        'latent_dim': 256,
        'encoder': 'transformer',  # Options: 'cnn_lstm', 'transformer'
    },
    'training': {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 1e-4,
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau',
    },
    'paths': {
        'data_dir': 'data/processed',
        'model_dir': 'models/checkpoints',
        'log_dir': 'src/logs/training_logs',
    },
    'curriculum': {
        'use_curriculum': True,
        'stages': 5,
    },
    'transfer_learning': {
        'freeze_layers': True,
        'layers_to_freeze': ['encoder'],
    }
})