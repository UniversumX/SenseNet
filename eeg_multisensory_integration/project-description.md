# SenseNet: Neural Cross-Modal Integration Project

## Team Requirement

We will need 7+ people on this team. It is one of our higher priority projects and papers we are trying to write. You can work on this and another project if you would like. 

Resources (People): 7+


## Project Overview


SenseNet investigates how the brain integrates different types of sensory information by studying the relationship between proprioceptive and visual perception using EEG data. The project employs advanced machine learning techniques, particularly focusing on transfer learning and curriculum learning approaches.

## Core Hypothesis

The project tests the hypothesis that neural activity contains an integrated world model that facilitates transfer learning between different perceptual modalities (specifically proprioception and vision). This suggests that learning in one modality should transfer effectively to another due to shared underlying neural representations.

## Technical Approach

### Data Collection & Generation
- Initially using synthetic datasets due to data collection constraints
- Future empirical data collection planned for next semester
- Two primary data types:
  1. EEG signals during object manipulation with proprioceptive data both during the visual fixation and on its own. 
  2. EEG signals during visual fixation tasks both during the object manipulation and on its own. 

### Model Architecture
- Implements a HydraNet-style architecture with:
  - Shared encoder backbone for EEG data processing
  - Separate decoders for proprioceptive and visual tasks
  - Cross-modal attention mechanisms
  - Task-specific heads with gating networks

### Learning Strategy
1. **Curriculum Learning Implementation**
   - Start: Simple movement predictions (gyroscopic data)
   - Progress: Complex 3D joint positions
   - Visual Features: Begin with basic features, advance to complex attributes
   - Gradually increase task difficulty and prediction precision

2. **Transfer Learning Process**
   - Phase 1: Train encoder on proprioceptive prediction
   - Phase 2: Adapt encoder for visual feature prediction
   - Focus: Maintaining shared representations while specializing for each modality

## Experimental Protocol

### Object Manipulation Task
- Participants manipulate objects with varying properties:
  - Shapes, sizes, textures
  - Colors, edge features
  - Systematic variation of object attributes
- Data Collected:
  - EEG recordings
  - Arm position tracking
  - Movement trajectories

### Visual Fixation Task
- Participants observe objects:
  - With and without manipulation
  - Controlled viewing conditions
  - Systematic feature presentation
- Data Collected:
  - EEG recordings
  - Eye tracking data
  - Visual feature annotations

## Technical Implementation Details

### Data Processing Pipeline
1. EEG Signal Processing
   - Filtering and artifact removal
   - Channel selection and preprocessing
   - Temporal alignment with behavioral data

2. Feature Extraction
   - Proprioceptive feature computation
   - Visual feature extraction
   - Cross-modal synchronization

### Model Components
1. EEG Encoder
   - Temporal convolution layers
   - Spatial attention mechanisms
   - Robust to signal variations

2. Task-Specific Decoders
   - Proprioceptive trajectory prediction
   - Visual feature prediction
   - Adaptive gating mechanisms

## Validation and Analysis

### Model Validation
1. Quantitative Metrics
   - Prediction accuracy for both modalities
   - Transfer learning effectiveness
   - Cross-modal prediction performance

2. Interpretability Analysis
   - t-SNE visualization of latent spaces
   - Saliency mapping for feature importance
   - Attribution analysis for predictions

### Expected Outcomes
1. Technical Validation
   - Demonstration of effective transfer learning
   - Quantification of cross-modal dependencies
   - Performance benchmarks for both tasks

2. Theoretical Insights
   - Evidence for integrated world model
   - Understanding of cross-modal representations
   - Implications for perceptual processing

## Project Timeline and Milestones

### Phase 1: Development (Winter Break and Spring Semester)
- Synthetic data generation
- Model architecture implementation
- Initial testing and validation

### Phase 2: Empirical (Spring Semester, possibly Summer or Fall)
- Data collection setup
- Human subject protocols
- Empirical data collection
- Model refinement

## Technical Requirements

### Software Dependencies
- Python 3.8+
- PyTorch for deep learning
- MNE for EEG processing
- Additional libraries specified in requirements.txt

### Hardware Requirements
- GPU-enabled computing environment (provided by the Universum Team and Neurotech Grants, external Grants/Venture Capital)
- EEG recording equipment (Phase 2, provided by UIUC likely)
- Motion tracking system (Phase 2, provided by team in UIUC likely)

## Getting Started

### Setup Instructions
1. Clone the repository
2. Install dependencies from requirements.txt
3. Run/Refine synthetic data generation scripts
4. Execute training pipeline
5. Evaluate results and set up the Analysis in the Notebooks

### Code Structure
- `/src`: Core implementation
- `/data`: Data handling and processing
- `/models`: Neural network architectures
- `/analysis`: Evaluation and visualization
- `/docs`: Documentation and protocols

## Future Directions

### Planned Extensions
1. Additional modalities (auditory, tactile)
2. Real-time processing capabilities
3. Advanced interpretation methods

### Applications
1. Brain-computer interfaces
2. Perceptual modeling
3. Neural engineering

## Contributing

### Guidelines
1. Code style: PEP 8
2. Documentation: Google style docstrings
3. Testing: PyTest for unit tests
4. Version control: Git flow workflow

### Review Process
1. Code review requirements
2. Testing standards
3. Documentation updates

For questions or contributions, please contact the project maintainers.




## Literature Review

Tesla’s Full Self-Driving (FSD) system employs a neural network architecture known as HydraNet to process visual data from its vehicles’ cameras. HydraNet features a shared backbone that extracts features from input images, which are then utilized by multiple task-specific branches, or “heads,” to perform various perception tasks such as object detection, lane recognition, and depth estimation. This design allows for efficient computation and resource sharing among tasks, enhancing the system’s overall performance. ￼

In parallel, research in neuroscience has explored the relationship between visual perception and proprioception—the sense of body position and movement—using electroencephalography (EEG). Studies have investigated how the brain integrates visual and proprioceptive feedback during motor tasks, revealing distinct and additive effects on cortical activity. For instance, combining visual feedback with vibratory proprioceptive stimulation during motor imagery tasks has been shown to modulate EEG patterns, suggesting that the brain dynamically integrates multiple sensory inputs to enhance motor control. ￼

While both Tesla’s HydraNet and EEG studies involve processing visual information, they operate in different domains and serve distinct purposes. HydraNet is engineered for autonomous vehicle perception, focusing on interpreting external visual data to navigate environments safely. In contrast, EEG research aims to understand how the human brain processes and integrates sensory information, including visual and proprioceptive cues, to inform motor actions and perception.

Despite these differences, there is a conceptual parallel in the integration of multiple data sources. Tesla’s system combines inputs from various cameras and sensors to create a comprehensive understanding of the vehicle’s surroundings, akin to how the human brain integrates sensory information to form a coherent perception of the environment. Both systems highlight the importance of multisensory integration in complex perception tasks, whether in artificial intelligence applications or human cognition.

Please add your own resources that you find, try finding 3+ resources of your own and add it with a summary of how it is related and what it is. 

To gain a comprehensive understanding of the integration of EEG-based proprioceptive and visual perception through transfer learning, as well as Tesla’s HydraNet architecture for autonomous driving, consider reviewing the following research papers and resources:

1. Transfer Learning in EEG-Based Brain-Computer Interfaces (BCIs):
	•	“Transfer Learning for EEG-Based Brain–Computer Interfaces: A Review of Progress Made Since 2016” by Dongrui Wu, Yifan Xu, and Bao-Liang Lu. This paper provides an extensive review of transfer learning approaches in EEG-based BCIs, covering various paradigms and applications, including motor imagery and event-related potentials. ￼
	•	“Transfer learning-based EEG analysis of visual attention and working memory”. This research explores the impact of cognitive activities on brain waves in the motor cortex, utilizing transfer learning for EEG data analysis. ￼
	•	“A Novel Deep Transfer Learning Framework Integrating General and Domain-Specific Features for EEG-Based Brain-Computer Interface” by Zilin Liang et al. This study introduces a deep transfer learning framework that combines general and domain-specific features for EEG signal processing in BCIs. ￼

2. Cross-Modal Integration of Proprioceptive and Visual Perception:
	•	“Cogni-Net: Cognitive Feature Learning through Deep Visual Perception” by Pranay Mukherjee et al. This paper proposes a model that leverages visual domain knowledge to train a recurrent model on brain signals, aiming to learn a discriminative manifold of human brain cognition in response to visual cues. ￼
	•	“See What You See: Self-supervised Cross-modal Retrieval of Visual Stimuli from Brain Activity” by Zesheng Ye et al. This study presents a self-supervised approach to retrieve visual stimuli from EEG data, focusing on cross-modal alignment between brain activity and visual inputs. ￼

3. Tesla’s HydraNet Architecture for Autonomous Driving:
	•	“Tesla’s HydraNet - How Tesla’s Autopilot Works”. This article provides an in-depth explanation of Tesla’s HydraNet, a multi-task learning architecture that processes inputs from multiple cameras to perform various perception tasks simultaneously. ￼
	•	“Breakdown: How Tesla will transition from Modular to End-To-End Deep Learning”. This piece discusses Tesla’s shift from a modular approach to an end-to-end deep learning system, highlighting the role of HydraNet in this transition. ￼
	•	“Tesla’s Transition to HydraNet: A Technical Deep Dive”. This article delves into the technical aspects of Tesla’s adoption of HydraNet for perception tasks in autonomous driving. ￼

These resources offer a solid foundation for understanding the application of transfer learning in EEG-based studies of proprioception and visual perception, as well as the architecture and functionality of Tesla’s HydraNet in autonomous vehicle perception systems.

