"""
Retrain two Poisson VAE models:
1. poisson_temp0.1_annealTrue_n256_seed0_v2 - using poisson_lin_lin config but with temp=0.1, anneal=true, n_exp=256
2. poisson_lin_lin_v2 - exact replica of poisson_lin_lin training

Usage:
    python retrain_models.py --model <model_name> --gpu <gpu_id>
"""

import argparse
import json
import torch
from pathlib import Path
from main.vae import PoissonVAE
from main.config_vae import ConfigPoisVAE, ConfigTrainVAE
from base.dataset import make_dataset
from main.train_vae import TrainerVAE


def load_config(config_path):
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config_dict, save_path):
    """Save config to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def train_model_v2(model_name, gpu_id):
    """
    Train a model based on poisson_lin_lin configs.
    
    Args:
        model_name: Either 'sweep_anneal' or 'linlin_v2'
        gpu_id: GPU ID to use for training
    """
    
    # Set device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base configs from poisson_lin_lin
    base_model_config = load_config('./checkpoints/poisson_lin_lin/ConfigPoisVAE.json')
    base_train_config = load_config('./checkpoints/poisson_lin_lin/ConfigTrainVAE.json')
    
    if model_name == 'sweep_anneal':
        # Modify config for sweep_anneal version
        save_dir = './checkpoints/poisson_temp0.1_annealTrue_n256_seed0_v2'
        
        # Modify epochs to 1500 instead of 3000
        base_train_config['epochs'] = 1500
        # Adjust T_max for cosine scheduler accordingly
        base_train_config['scheduler_kws']['T_max'] = 1495.0
        
        print("=" * 70)
        print("Training: poisson_temp0.1_annealTrue_n256_seed0_v2")
        print("Config: Based on poisson_lin_lin (temp anneal 1.0->0.05)")
        print("Modified: epochs=1500 (instead of 3000)")
        print("=" * 70)
        
    elif model_name == 'linlin_v2':
        # Exact replica of poisson_lin_lin
        save_dir = './checkpoints/poisson_lin_lin_v2'
        
        print("=" * 70)
        print("Training: poisson_lin_lin_v2")
        print("Config: Exact replica of poisson_lin_lin")
        print("=" * 70)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configs to new directory
    save_config(base_model_config, Path(save_dir) / 'ConfigPoisVAE.json')
    save_config(base_train_config, Path(save_dir) / 'ConfigTrainVAE.json')
    
    # Create model config
    model_cfg = ConfigPoisVAE(dataset=base_model_config['dataset'])
    for k, v in base_model_config.items():
        if hasattr(model_cfg, k) and k != 'dataset':
            setattr(model_cfg, k, v)
    
    # Create training config
    train_cfg = ConfigTrainVAE()
    for k, v in base_train_config.items():
        if hasattr(train_cfg, k):
            setattr(train_cfg, k, v)
    
    # Set save directory
    train_cfg.save_dir = save_dir
    
    # Create model
    print("\nCreating model...")
    model = PoissonVAE(model_cfg)
    model = model.to(device)
    
    print(f"Model config:")
    print(f"  Encoder: {model_cfg.enc_type}, Decoder: {model_cfg.dec_type}")
    print(f"  N latents: {model_cfg.n_latents}")
    print(f"  Dataset: {model_cfg.dataset}")
    
    print(f"\nTraining config:")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.lr}")
    print(f"  Temp anneal: {train_cfg.temp_start} -> {train_cfg.temp_stop} (portion: {train_cfg.temp_anneal_portion})")
    
    # Create trainer (it will load dataset automatically in setup_data)
    print("\nCreating trainer...")
    trainer = TrainerVAE(
        model=model,
        cfg=train_cfg,
        device=str(device),
        verbose=True,
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 70)
    
    # Get the fit_name from the save_dir
    fit_name = Path(save_dir).name
    trainer.train(fit_name=fit_name, save=True)
    
    print("\n" + "=" * 70)
    print(f"Training complete! Model saved to: {trainer.model.chkpt_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain Poisson VAE models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['sweep_anneal', 'linlin_v2'],
                        help='Which model to train')
    parser.add_argument('--gpu', type=int, required=True,
                        help='GPU ID to use for training')
    
    args = parser.parse_args()
    
    train_model_v2(args.model, args.gpu)
