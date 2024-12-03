import torch
import wandb
from eeg_hydranet import IntegratedWorldSimulator, DataConfig
from models import EEGHydraNet, EEGConfig
from hydranet_experiment import run_experiment

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Initialize wandb
    wandb.login()
    
    # Run the experiment
    model, metrics = run_experiment()
    
    print("\nExperiment completed!")
    print("\nTest Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': metrics,
        'config': model.config
    }, 'final_model.pt')
