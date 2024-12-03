import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from eeg_hydranet import EEGConfig, EEGHydraNet, EEGDataset
from data_generation.synth_data import IntegratedWorldSimulator, DataConfig

def run_experiment():
    # 1. Configuration
    config = EEGConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Generate synthetic data
    print("Generating synthetic data...")
    simulator = IntegratedWorldSimulator(DataConfig())
    dataset_dict = simulator.generate_integrated_dataset()
    
    # 3. Create dataset and splits
    print("Creating datasets...")
    full_dataset = EEGDataset(dataset_dict, config)
    
    # Calculate splits
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 5. Initialize model and training components
    print("Initializing model...")
    model = EEGHydraNet(config).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=5,
        factor=0.5
    )
    
    # Loss functions
    proprio_criterion = nn.MSELoss()
    visual_criterion = nn.MSELoss()
    
    # 6. Initialize wandb
    wandb.init(
        project="perception-hydranet",
        config={
            "architecture": "hydranet",
            "dataset": "synthetic",
            **vars(config)
        }
    )
    
    # 7. Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    early_stopping_patience = 10
    no_improve_count = 0
    
    for epoch in range(config.n_epochs):
        # Training
        model.train()
        train_losses = {
            'proprio': [],
            'visual': [],
            'total': []
        }
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.n_epochs}'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['eeg'].size(0)
            seq_len = batch['eeg'].size(2)  # This should now be n_timepoints-1
            
            # Forward pass
            outputs, gate_scores = model(batch)
            
            # Process proprioception targets
            # proprio_targets is already in shape (batch_size, seq_len, n_joints*3)
            proprio_loss = proprio_criterion(
                outputs[0],  # Shape: (batch_size, seq_len, n_joints*3)
                batch['proprio_targets']  # Shape: (batch_size, seq_len, n_joints*3)
            )
            
            # Process visual targets
            # visual_targets is already in shape (batch_size, seq_len, n_visual_features)
            visual_loss = visual_criterion(
                outputs[1],  # Shape: (batch_size, seq_len, n_visual_features)
                batch['visual_targets']  # Shape: (batch_size, seq_len, n_visual_features)
            )
            
            
            # Total loss
            total_loss = (
                config.task_weights['proprioception'] * proprio_loss +
                config.task_weights['visual'] * visual_loss
            )
            
            # Add gating regularization
            gate_loss = config.gate_sparsity_weight * (
                gate_scores.mean() - config.target_sparsity
            ).abs()
            total_loss += gate_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            train_losses['proprio'].append(proprio_loss.item())
            train_losses['visual'].append(visual_loss.item())
            train_losses['total'].append(total_loss.item())
        
        # Validation
        model.eval()
        val_losses = {
            'proprio': [],
            'visual': [],
            'total': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size = batch['eeg'].size(0)
                seq_len = batch['eeg'].size(2)
                
                outputs, gate_scores = model(batch)
                
                proprio_targets = batch['proprio_targets'].reshape(batch_size, seq_len, -1)
                proprio_outputs = outputs[0]
                proprio_loss = proprio_criterion(
                    proprio_outputs,
                    proprio_targets
                )
                
                visual_targets = batch['visual_targets']
                visual_outputs = outputs[1]
                visual_loss = visual_criterion(
                    visual_outputs,
                    visual_targets
                )
                
                total_loss = (
                    config.task_weights['proprioception'] * proprio_loss +
                    config.task_weights['visual'] * visual_loss
                )
                
                val_losses['proprio'].append(proprio_loss.item())
                val_losses['visual'].append(visual_loss.item())
                val_losses['total'].append(total_loss.item())
        
        # Calculate metrics
        train_metrics = {
            f'train_{k}_loss': np.mean(v) 
            for k, v in train_losses.items()
        }
        
        val_metrics = {
            f'val_{k}_loss': np.mean(v) 
            for k, v in val_losses.items()
        }
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            **train_metrics,
            **val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config.n_epochs}")
        print(f"Train Loss: {train_metrics['train_total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_total_loss']:.4f}")
        
        # Learning rate scheduling
        current_val_loss = val_metrics['val_total_loss']
        scheduler.step(current_val_loss)
        
        # Early stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            no_improve_count = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, 'best_model.pt')
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print("\nEarly stopping triggered!")
                break
    
    # 8. Final evaluation on test set
    print("\nRunning final evaluation...")
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_losses = {
        'proprio': [],
        'visual': [],
        'total': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size = batch['eeg'].size(0)
            seq_len = batch['eeg'].size(2)
            
            outputs, gate_scores = model(batch)
            
            proprio_targets = batch['proprio_targets'].reshape(batch_size, seq_len, -1)
            proprio_outputs = outputs[0]
            proprio_loss = proprio_criterion(
                proprio_outputs, proprio_targets
            )
            
            visual_targets = batch['visual_targets']
            visual_outputs = outputs[1]
            visual_loss = visual_criterion(
                visual_outputs, visual_targets
            )
            
            total_loss = (
                config.task_weights['proprioception'] * proprio_loss +
                config.task_weights['visual'] * visual_loss
            )
            
            test_losses['proprio'].append(proprio_loss.item())
            test_losses['visual'].append(visual_loss.item())
            test_losses['total'].append(total_loss.item())
    
    test_metrics = {
        f'test_{k}_loss': np.mean(v) 
        for k, v in test_losses.items()
    }
    
    # Log final test metrics
    wandb.log(test_metrics)
    
    # 9. Generate visualizations
    plot_results(train_losses, val_losses, test_metrics)
    
    return model, test_metrics

def plot_results(train_losses, val_losses, test_metrics):
    """Generate and save visualization plots"""
    plt.figure(figsize=(15, 5))
    
    # Training curves
    plt.subplot(1, 3, 1)
    plt.plot([np.mean(train_losses['total'][i:i+10]) 
              for i in range(0, len(train_losses['total']), 10)],
             label='Train Loss')
    plt.plot([np.mean(val_losses['total'][i:i+10]) 
              for i in range(0, len(val_losses['total']), 10)],
             label='Val Loss')
    plt.title('Training Progress')
    plt.xlabel('Step (x10)')
    plt.ylabel('Loss')
    plt.legend()
    
    # Task-specific training curves
    plt.subplot(1, 3, 2)
    for task in ['proprio', 'visual']:
        plt.plot([np.mean(train_losses[task][i:i+10]) 
                 for i in range(0, len(train_losses[task]), 10)],
                 label=f'{task} Loss')
    plt.title('Task-Specific Training Progress')
    plt.xlabel('Step (x10)')
    plt.ylabel('Loss')
    plt.legend()
    
    # Final test performance
    plt.subplot(1, 3, 3)
    tasks = ['proprio', 'visual']
    test_scores = [test_metrics[f'test_{task}_loss'] for task in tasks]
    plt.bar(tasks, test_scores)
    plt.title('Test Performance by Task')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    wandb.log({"training_curves": wandb.Image('training_results.png')})

if __name__ == "__main__":
    model, metrics = run_experiment()
    print("\nFinal Test Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")