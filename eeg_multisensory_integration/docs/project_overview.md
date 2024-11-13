Table of Contents

	1.	Project Overview
	2.	High-Level Workflow
	3.	Project Structure
	•	Root Directory
	•	Data Directory
	•	Notebooks Directory
	•	Source Code (src/) Directory
	•	Documentation (docs/) Directory
	•	Scripts Directory
	4.	Detailed Explanation of Each Component
	•	Configuration (config.py)
	•	Data Generation
	•	Data Preprocessing
	•	Data Synchronization
	•	Model Architectures
	•	Training Pipeline
	•	Evaluation and Interpretability
	•	Utilities
	5.	Flow of Data and Processes
	•	Step 1: Data Generation
	•	Step 2: Data Preprocessing
	•	Step 3: Data Synchronization
	•	Step 4: Model Training
	•	Step 5: Transfer Learning
	•	Step 6: Model Evaluation
	•	Step 7: Comparative Analysis
	6.	Advanced Techniques and Methodologies
	•	Curriculum Learning
	•	Transfer Learning
	•	Transformer-based EEG Encoder
	•	Interpretability Methods
	7.	Modifiability and Extensibility
	8.	Conclusion

Project Overview

The EEG Multisensory Integration Project aims to investigate the shared neural representations underlying proprioceptive and visual perception using EEG data. The project leverages:
	•	Curriculum Learning: Gradually increasing task complexity during training.
	•	Transfer Learning: Adapting a model trained on one task to perform another related task.
	•	Synthetic Data Generation: Creating realistic synthetic datasets due to current limitations in data collection.
	•	Advanced Neural Network Architectures: Utilizing state-of-the-art models suitable for EEG data, such as Transformers.
	•	Interpretability Techniques: Understanding what features the models learn and how they make predictions.

The ultimate goal is to demonstrate that transfer learning between proprioceptive and visual modalities is facilitated by an integrated perceptual world model encoded in neural activity.

High-Level Workflow

	1.	Data Generation: Create synthetic EEG, proprioceptive, and visual datasets that simulate realistic scenarios.
	2.	Data Preprocessing: Clean and prepare the data for modeling, including filtering, artifact removal, and normalization.
	3.	Data Synchronization: Align the different modalities temporally to ensure that they correspond correctly.
	4.	Model Training:
	•	Stage 1: Train a shared encoder and a proprioceptive decoder on the combined EEG and proprioceptive data using curriculum learning.
	•	Stage 2: Transfer the encoder to predict visual features from EEG data, training a visual decoder.
	5.	Evaluation and Interpretability: Assess model performance and interpret the learned representations using various techniques.
	6.	Comparative Analysis: Compare different models to validate the hypothesis about integrated perceptual representations.

Project Structure

Root Directory

eeg_multisensory_integration/
├── README.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
├── docs/
└── scripts/

	•	README.md: Provides an overview of the project, setup instructions, and how to run the code.
	•	requirements.txt: Lists all Python dependencies required for the project.

Data Directory

data/
├── raw/
├── processed/
├── synthetic/
└── datasets_info.txt

	•	raw/: Placeholder for raw datasets (when available).
	•	processed/: Preprocessed data ready for modeling.
	•	synthetic/: Synthetic datasets generated for the project.
	•	datasets_info.txt: Information about datasets used.

Notebooks Directory

notebooks/
├── exploratory_data_analysis.ipynb
├── model_training.ipynb
└── results_analysis.ipynb

	•	exploratory_data_analysis.ipynb: Notebook for exploring and visualizing the data.
	•	model_training.ipynb: Interactive notebook for training models and experimenting with parameters.
	•	results_analysis.ipynb: Analyzes the results, visualizes performance, and interprets the models.

Source Code (src/) Directory

src/
├── config.py
├── data_generation/
├── data_preprocessing/
├── data_synchronization/
├── models/
├── training/
├── evaluation/
├── utils/
└── logs/

Key Subdirectories and Files:

	•	config.py: Centralized configuration file for managing hyperparameters and paths.
	•	data_generation/: Scripts for generating synthetic data.
	•	data_preprocessing/: Scripts for preprocessing the data.
	•	data_synchronization/: Scripts for aligning the data across modalities.
	•	models/: Model architectures, including the EEG encoder and decoders.
	•	training/: Training pipelines, including curriculum and transfer learning implementations.
	•	evaluation/: Scripts for evaluating models and interpreting results.
	•	utils/: Utility functions for data loading, metrics, and visualization.
	•	logs/: Directory for storing training logs.

Documentation (docs/) Directory

docs/
├── literature_review.md
├── experimental_protocol.md
├── ethics_application.md
└── equipment_plan.md

	•	Contains documentation for literature reviews, experimental protocols, ethics applications, and equipment planning.

Scripts Directory

scripts/
├── run_data_generation.sh
├── run_preprocessing.sh
├── run_training.sh
├── run_evaluation.sh
└── run_comparative_analysis.sh

	•	Shell scripts to execute various parts of the project pipeline conveniently.

Detailed Explanation of Each Component

Configuration (config.py)

The config.py file uses the Hydra and OmegaConf libraries to create a centralized configuration. This allows easy management of hyperparameters, file paths, and model settings.

# config.py
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'data': {
        'eeg_channels': 64,
        'sampling_rate': 256,
        'sequence_length': 1024,
        'n_samples': 10000,
        'visual_feature_dim': 20,  # Number of visual features
        'n_joints': 7,  # Number of joints in proprioceptive data
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

Purpose:
	•	Modularity: Changing parameters in one place updates them throughout the project.
	•	Experimentation: Facilitates easy experimentation with different settings.
	•	Reproducibility: Ensures that experiments can be replicated with the same configurations.

Data Generation

Overview

Due to limitations in collecting real data, the project generates synthetic datasets that simulate EEG signals, proprioceptive data, and visual features.

Components

	•	generate_synthetic_eeg.py: Generates realistic synthetic EEG data using MNE’s simulation tools and signal processing techniques.
	•	generate_synthetic_proprioceptive.py: Simulates proprioceptive data representing realistic arm movements using kinematic models.
	•	generate_synthetic_visual.py: Creates synthetic visual feature data representing objects with varying attributes.

Key Techniques

	•	EEG Simulation: Combines multiple sine waves with random frequencies and phases to mimic EEG rhythms, adding Gaussian noise to simulate real-world data.
	•	Proprioceptive Simulation: Uses sine waves with low frequencies to represent slow, natural arm movements across different joints.
	•	Visual Feature Generation: Randomly selects features (edges, shapes, colors, sizes, textures) and encodes them using one-hot encoding for use in models.

Data Preprocessing

Overview

Preprocessing is crucial to clean the data and prepare it for modeling, ensuring that the models receive high-quality inputs.

Components

	•	preprocess_eeg.py: Applies advanced EEG preprocessing steps, including filtering, artifact removal using Independent Component Analysis (ICA), and normalization.
	•	preprocess_proprioceptive.py: Smooths proprioceptive data using Savitzky-Golay filters, normalizes the data, and optionally extracts features.
	•	preprocess_visual.py: Normalizes visual feature data to ensure consistent scaling.

Key Techniques

	•	Filtering: Band-pass filters EEG data between 1-40 Hz to focus on relevant neural signals.
	•	Artifact Removal: Uses ICA to remove artifacts like eye blinks and muscle movements from EEG data.
	•	Normalization: Standardizes data to have zero mean and unit variance, improving model training stability.

Data Synchronization

Overview

Synchronizing data across modalities ensures that the EEG signals, proprioceptive data, and visual features correspond to the same time points and events.

Component

	•	synchronize_data.py: Aligns the datasets based on shared time vectors or indices, ensuring that each sample across modalities corresponds correctly.

Key Techniques

	•	Time Alignment: If data has timestamps, synchronization ensures that data points from different modalities align temporally.
	•	Index Alignment: In synthetic data, we can assume alignment based on sample indices.

Model Architectures

Overview

The project employs advanced neural network architectures suitable for processing EEG data and modeling complex relationships between modalities.

Components

	•	eeg_encoder.py: Contains a function to select and instantiate the appropriate EEG encoder based on the configuration.
	•	transformer_encoder.py: Implements a Transformer-based encoder for EEG data.
	•	proprioceptive_decoder.py and visual_decoder.py: Implement decoders to predict proprioceptive and visual data from the latent representations generated by the encoder.

Key Techniques

	•	Transformer Encoder: Leverages self-attention mechanisms to capture temporal dependencies in EEG data more effectively than traditional RNNs.
	•	Flexible Architecture: Allows selection between different encoders (e.g., CNN-LSTM, Transformer) based on experimentation needs.
	•	Deep Decoders: Use multiple layers, activation functions, and regularization techniques like Batch Normalization to improve learning.

Training Pipeline

Overview

The training pipeline orchestrates the model training process, including curriculum learning and transfer learning stages.

Components

	•	train_model.py: Main script for training models, handling both proprioceptive and visual tasks.
	•	curriculum_learning.py: Implements curriculum learning strategies to gradually increase data complexity.
	•	transfer_learning.py: Sets up models for transfer learning, handling layer freezing and optimizer adjustments.

Key Techniques

	•	Curriculum Learning: Gradually introduces more complex data during training to improve learning efficiency and model performance.
	•	Transfer Learning: Transfers knowledge from the encoder trained on proprioceptive tasks to visual tasks, hypothesizing that shared representations exist.
	•	Advanced Optimizers and Schedulers: Uses optimizers like AdamW and learning rate schedulers to enhance training.

Logging and Monitoring

	•	TensorBoard Integration: Logs training metrics, losses, and model graphs for visualization and monitoring.
	•	Training Logs: Stores logs in the logs/ directory for later analysis.

Evaluation and Interpretability

Overview

Evaluation assesses model performance using various metrics and interpretability techniques to understand the learned representations.

Components

	•	evaluate_model.py: Evaluates the trained models on test data, computing performance metrics.
	•	interpretability.py: Applies techniques like t-SNE for latent space visualization and saliency maps for feature importance.
	•	comparative_analysis.py: Compares different models to validate hypotheses about integrated perceptual representations.

Key Techniques

	•	Performance Metrics: Uses metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score to assess regression tasks.
	•	t-SNE Visualization: Reduces high-dimensional latent representations to 2D for visualization, helping to interpret clustering and separability.
	•	Saliency Maps: Uses libraries like Captum to compute gradients with respect to inputs, highlighting important features.

Utilities

Overview

Utility functions support data loading, metric computations, and visualizations, making the codebase cleaner and more maintainable.

Components

	•	data_loader.py: Contains classes and functions to load data into PyTorch datasets and dataloaders.
	•	metrics.py: Computes performance metrics for regression tasks.
	•	visualization.py: Provides functions for plotting and visualizing data and results.

Key Techniques

	•	Efficient Data Loading: Utilizes PyTorch’s DataLoader with optimizations like multiprocessing (num_workers) and pinned memory.
	•	Data Augmentation (Future Work): Placeholder for implementing data augmentation techniques if needed.

Flow of Data and Processes

Step 1: Data Generation

	•	Scripts:
	•	generate_synthetic_eeg.py
	•	generate_synthetic_proprioceptive.py
	•	generate_synthetic_visual.py
	•	Process:
	1.	Generate synthetic EEG data simulating neural activity during object manipulation and visual fixation tasks.
	2.	Generate synthetic proprioceptive data representing realistic arm movements.
	3.	Generate synthetic visual feature data representing objects with varying visual attributes.
	•	Output:
	•	Synthetic datasets saved in data/synthetic/.

Step 2: Data Preprocessing

	•	Scripts:
	•	preprocess_eeg.py
	•	preprocess_proprioceptive.py
	•	preprocess_visual.py
	•	Process:
	1.	Clean EEG data by filtering, removing artifacts, and normalizing.
	2.	Smooth and normalize proprioceptive data.
	3.	Normalize visual feature data.
	•	Output:
	•	Preprocessed data saved in data/processed/.

Step 3: Data Synchronization

	•	Script:
	•	synchronize_data.py
	•	Process:
	•	Align EEG, proprioceptive, and visual data based on shared indices or time vectors.
	•	Output:
	•	Synchronized datasets saved in data/processed/.

Step 4: Model Training

	•	Script:
	•	train_model.py
	•	Process:
	•	Stage 1: Train the shared encoder and proprioceptive decoder using curriculum learning.
	•	Gradually introduce more complex data (e.g., from simple gyroscopic data to complex joint positions).
	•	Logging: Use TensorBoard to monitor training progress.
	•	Output:
	•	Trained encoder and decoder models saved in models/checkpoints/.

Step 5: Transfer Learning

	•	Script:
	•	train_model.py (with transfer learning configurations)
	•	Process:
	•	Transfer the pretrained encoder to the visual task.
	•	Freeze specified layers if necessary.
	•	Train the visual decoder to predict visual features from EEG data.
	•	Output:
	•	Updated encoder and trained visual decoder models saved in models/checkpoints/.

Step 6: Model Evaluation

	•	Script:
	•	evaluate_model.py
	•	Process:
	•	Evaluate the performance of the trained models on test datasets.
	•	Compute performance metrics and save results.
	•	Output:
	•	Evaluation metrics printed and saved for analysis.

Step 7: Comparative Analysis

	•	Script:
	•	comparative_analysis.py
	•	Process:
	•	Compare different models (e.g., integrated training vs. sequential training vs. random initialization) to test the hypothesis about shared representations.
	•	Use statistical tests to assess significance.
	•	Output:
	•	Comparative results and visualizations saved for reporting.

Advanced Techniques and Methodologies

Curriculum Learning

	•	Purpose: Improves learning efficiency by starting with simpler tasks and progressively increasing complexity.
	•	Implementation:
	•	Data complexity increases over predefined stages.
	•	The get_curriculum_dataloader function adjusts the dataset for each stage.
	•	Benefits:
	•	Helps the model learn fundamental patterns before tackling more complex ones.
	•	Can lead to better generalization and faster convergence.

Transfer Learning

	•	Purpose: Leverages knowledge from one task (proprioceptive prediction) to improve performance on another related task (visual feature prediction).
	•	Implementation:
	•	Pretrained encoder’s weights are transferred to the new task.
	•	Layers can be frozen or fine-tuned based on the configuration.
	•	Benefits:
	•	Reduces the amount of data and time needed to train the model on the new task.
	•	Tests the hypothesis that the encoder captures shared representations.

Transformer-based EEG Encoder

	•	Purpose: Captures temporal dependencies in EEG data more effectively using self-attention mechanisms.
	•	Implementation:
	•	The TransformerEEGEncoder class implements a Transformer encoder architecture.
	•	Configurable parameters like the number of heads, layers, and hidden dimensions.
	•	Benefits:
	•	Handles long sequences better than RNNs.
	•	Can capture global relationships within the data.

Interpretability Methods

	•	t-SNE Visualization:
	•	Reduces high-dimensional latent representations to 2D.
	•	Helps visualize clusters and the structure of the latent space.
	•	Saliency Maps:
	•	Highlights which parts of the input data contribute most to the model’s predictions.
	•	Uses gradient-based methods from libraries like Captum.
	•	Representation Similarity Analysis (Future Work):
	•	Measures the similarity between representations learned from different modalities.

Modifiability and Extensibility

	•	Modular Design: Each component is self-contained, making it easy to modify or replace parts without affecting the whole system.
	•	Configuration Management: Centralized configurations allow easy tuning of hyperparameters and settings.
	•	Model Flexibility: The code supports different model architectures and can be extended to include new ones.
	•	Data Agnostic: While currently using synthetic data, the pipeline can be adapted to real datasets with minimal changes.
	•	Logging and Monitoring: Integrated logging facilitates debugging and performance tracking.
	•	Documentation: Comprehensive documentation and code comments aid in understanding and extending the codebase.

Conclusion

The EEG Multisensory Integration Project is a comprehensive effort to model and understand the shared neural representations between proprioceptive and visual perception using EEG data. The project employs state-of-the-art techniques in data generation, preprocessing, neural network architectures, curriculum learning, and transfer learning.

By structuring the project in a modular and extensible manner, the codebase is prepared for future developments, including the integration of empirical data and further experimentation with advanced models and techniques. The detailed explanation provided here should help team members and contributors understand the flow of data and processes, facilitating collaboration and innovation within the project.

Next Steps:
	•	Integration with Real Data: When empirical data becomes available, the synthetic data components can be replaced or supplemented accordingly.
	•	Experimentation with Different Architectures: Try alternative models like Graph Neural Networks (GNNs) if suitable.
	•	Advanced Interpretability: Implement additional techniques like layer-wise relevance propagation (LRP) for deeper insights.
	•	Hyperparameter Optimization: Use tools like Optuna for automated hyperparameter tuning.
	•	Deployment Considerations: Prepare the models for deployment in practical applications, considering optimization and scalability.

