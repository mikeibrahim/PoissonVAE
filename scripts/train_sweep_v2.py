#!/usr/bin/env python3
"""
Training script for parameter sweep experiments (v2 - corrected).
This version uses the proper training approach with dynamic n_exp updates.
Usage: python scripts/train_sweep_v2.py --config CONFIG_JSON --gpu GPU_ID
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from main.config_vae import ConfigPoisVAE, ConfigTrainVAE
from main.vae import PoissonVAE
from main.train_vae import TrainerVAE
from base.dataset import make_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train P-VAE with sweep config (v2)')
    parser.add_argument('--config', type=str, required=True, help='JSON config string or file path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    return parser.parse_args()


def train_model(config: dict, gpu_id: int):
    """Train a single model with given configuration."""
    
    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Extract config
    dist_type = config['dist_type']  # 'poisson' or 'gumbel'
    temp = config['temp']
    anneal = config['anneal']
    n_param = config['n_param']  # n_exp for poisson, upperbound for gumbel
    seed = config['seed']
    epochs = config.get('epochs', 2000)
    
    # Create checkpoint path
    checkpoint_dir = Path('./checkpoints/sweep_2')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Model name
    anneal_str = 'annealTrue' if anneal else 'annealFalse'
    model_name = f"{dist_type}_temp{temp}_{anneal_str}_n{n_param}_seed{seed}"
    checkpoint_path = checkpoint_dir / model_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    print(f"  dist_type: {dist_type}")
    print(f"  temp: {temp}")
    print(f"  anneal: {anneal}")
    print(f"  n_param: {n_param}")
    print(f"  seed: {seed}")
    print(f"  epochs: {epochs}")
    print(f"  save_dir: {checkpoint_path}")
    
    # Load base configs from poisson_lin_lin
    base_config_dir = Path('./checkpoints/poisson_lin_lin')
    with open(base_config_dir / 'ConfigPoisVAE.json', 'r') as f:
        base_model_config = json.load(f)
    
    with open(base_config_dir / 'ConfigTrainVAE.json', 'r') as f:
        base_train_config = json.load(f)
    
    # Modify model config for sweep parameters
    base_model_config['seed'] = seed
    base_model_config['dist_type'] = dist_type
    
    if dist_type == 'gumbel':
        base_model_config['upperbound'] = n_param
    
    # Modify training config
    base_train_config['epochs'] = epochs
    base_train_config['scheduler_kws']['T_max'] = float(epochs - 5)
    
    # Set temperature annealing
    if anneal:
        base_train_config['temp_start'] = max(temp, 1.0)  # Start from at least 1.0 or temp
        base_train_config['temp_stop'] = temp * 0.05  # End at 5% of temp
        base_train_config['temp_anneal_portion'] = 0.5
    else:
        base_train_config['temp_start'] = temp
        base_train_config['temp_stop'] = temp
        base_train_config['temp_anneal_portion'] = 0.0
    
    # Save checkpoint at end only
    base_train_config['chkpt_freq'] = epochs
    
    # Save configs
    with open(checkpoint_path / 'ConfigPoisVAE.json', 'w') as f:
        json.dump(base_model_config, f, indent=2)
    
    with open(checkpoint_path / 'ConfigTrainVAE.json', 'w') as f:
        json.dump(base_train_config, f, indent=2)
    
    # Save sweep config
    with open(checkpoint_path / 'sweep_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model config object
    model_cfg = ConfigPoisVAE(dataset=base_model_config['dataset'])
    for k, v in base_model_config.items():
        if hasattr(model_cfg, k) and k != 'dataset':
            setattr(model_cfg, k, v)
    
    # Create training config object
    train_cfg = ConfigTrainVAE()
    for k, v in base_train_config.items():
        if hasattr(train_cfg, k):
            setattr(train_cfg, k, v)
    
    # Load dataset
    print("\nLoading dataset...")
    trn_data, vld_data, tst_data = make_dataset(
        dataset=model_cfg.dataset,
        device=device,
        load_dir="Datasets",
    )
    print(f"  Train: {len(trn_data)}, Val: {len(vld_data)}")
    
    # Create model
    print("\nCreating model...")
    model = PoissonVAE(model_cfg)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = TrainerVAE(
        model=model,
        cfg=train_cfg,
        device=str(device),
        verbose=True,
    )
    
    # Override checkpoint directory
    def create_chkpt_dir(self, fit_name=None):
        self.chkpt_dir = str(checkpoint_path)
        os.makedirs(self.chkpt_dir, exist_ok=True)
    
    model.create_chkpt_dir = lambda fit_name=None: create_chkpt_dir(model, fit_name)
    
    # Train
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        trainer.train(fit_name=model_name, save=True)
        
        train_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete! Time: {train_time/60:.1f} minutes")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*70}")
        
        # Compute final metrics
        print("\nComputing final metrics...")
        model.eval()
        with torch.no_grad():
            # Get validation batch
            from torch.utils.data import DataLoader
            dl_vld = DataLoader(vld_data, batch_size=1000, shuffle=False)
            x_val = next(iter(dl_vld))[0].to(device)
            
            # Forward pass
            dist, log_dr, spks, y = model(x_val)
            
            # Compute metrics
            kl_diag = model.loss_kl(log_dr)
            mse = torch.mean((x_val - y)**2, dim=[1,2,3])
            kl_per_sample = torch.sum(kl_diag, dim=1)
            recon_loss = torch.mean(mse) * (model.cfg.input_sz ** 2)
            kl_loss = torch.mean(kl_per_sample)
            neg_elbo = recon_loss + kl_loss
            
            # KL per dimension
            kl_per_dim = torch.mean(kl_diag, dim=0).cpu().numpy()
            n_active = int(np.sum(kl_per_dim > 0.01))
            
            # Lifetime sparsity
            spks_np = spks.cpu().numpy()
            active_per_neuron = np.mean(spks_np > 0, axis=0)
            lifetime_sparsity = float(np.mean(active_per_neuron))
        
        metrics = {
            'neg_elbo': float(neg_elbo.item()),
            'recon_loss': float(recon_loss.item()),
            'kl_loss': float(kl_loss.item()),
            'n_active': n_active,
            'lifetime_sparsity': lifetime_sparsity,
            'train_time': train_time,
            'kl_per_dim': kl_per_dim.tolist(),
        }
        
        # Save metrics
        with open(checkpoint_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nFinal Metrics:")
        print(f"  Neg ELBO: {neg_elbo.item():.2f}")
        print(f"  Recon Loss: {recon_loss.item():.2f}")
        print(f"  KL Loss: {kl_loss.item():.2f}")
        print(f"  Active neurons: {n_active}/512")
        print(f"  Lifetime sparsity: {lifetime_sparsity:.4f}")
        
        return True, metrics
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"Training FAILED: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    args = parse_args()
    
    # Load config
    if os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)
    
    # Train
    success, metrics = train_model(config, args.gpu)
    
    if success:
        print("\n✓ Training completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Training failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
