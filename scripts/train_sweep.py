#!/usr/bin/env python3
"""
Training script for parameter sweep experiments.
Usage: python scripts/train_sweep.py --config CONFIG_JSON
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
    parser = argparse.ArgumentParser(description='Train P-VAE with sweep config')
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
    checkpoint_dir = Path('./checkpoints/sweep')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = f"{dist_type}_temp{temp}_anneal{anneal}_n{n_param}_seed{seed}"
    checkpoint_path = checkpoint_dir / model_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"  Distribution: {dist_type}")
    print(f"  Temperature: {temp}")
    print(f"  Annealing: {anneal}")
    print(f"  N param: {n_param}")
    print(f"  Seed: {seed}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*80}\n")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Temperature annealing settings
    if anneal:
        temp_start = 1.0
        temp_stop = temp
        temp_anneal_portion = 0.5
    else:
        temp_start = temp
        temp_stop = temp
        temp_anneal_portion = 0.0
    
    # Model config
    model_cfg = ConfigPoisVAE(
        prior_clamp=-4,
        prior_log_dist="uniform",
        hard_fwd=False,
        exc_only=False,
        rmax_q=1.0,
        dataset="vH16",
        n_ch=32,
        n_latents=512,
        enc_type="lin",
        dec_type="lin",
        enc_bias=False,
        dec_bias=False,
        enc_norm=False,
        dec_norm=False,
        fit_prior=True,
        activation_fn="swish",
        init_dist="Normal",
        init_scale=0.0001,
        res_eps=1.0,
        use_bn=False,
        use_se=True,
        seed=seed,
        save=False,
        base_dir="./checkpoints",
        data_dir="Datasets",
    )
    model_cfg.mods_dir = str(checkpoint_path)
    
    # Add sweep-specific parameters to model config
    model_cfg.dist_type = dist_type
    if dist_type == 'gumbel':
        model_cfg.upperbound = n_param
        model_cfg.n_exp = 263  # default, not used
    else:
        model_cfg.n_exp = n_param
        model_cfg.upperbound = 5  # default, not used
    
    # Training config
    train_cfg = ConfigTrainVAE(
        method="mc",
        kl_beta=1.0,
        kl_beta_min=0.0001,
        kl_anneal_cycles=0,
        kl_anneal_portion=0.5,
        kl_const_portion=0.0,
        lambda_anneal=False,
        lambda_init=0.0,
        lambda_norm=0.0,
        temp_anneal_portion=temp_anneal_portion,
        temp_anneal_type="lin",
        temp_start=temp_start,
        temp_stop=temp_stop,
        lr=0.005,
        epochs=epochs,
        batch_size=1000,
        warm_restart=0,
        warmup_epochs=5,
        optimizer="adamax_fast",
        optimizer_kws={"weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-08},
        scheduler_type="cosine",
        scheduler_kws={"T_max": epochs - 5, "eta_min": 1e-05},
        ema_rate=None,
        grad_clip=500,
        use_amp=False,
        chkpt_freq=epochs,  # Save only final
        eval_freq=20,
        log_freq=10,
    )
    
    # Load dataset to CPU first, then move batches to GPU during training
    trn_data, vld_data, tst_data = make_dataset(
        dataset="vH16",
        device=torch.device('cpu'),  # Load to CPU to save GPU memory
        load_dir="Datasets",
    )
    
    # Initialize model
    model = PoissonVAE(model_cfg)
    model = model.to(device)
    
    # Update n_exp if using Poisson distribution
    if dist_type == 'poisson':
        model.n_exp.fill_(n_param)
    
    # Store extra config in model for later reference
    model.sweep_config = config
    
    # Override checkpoint directory
    def simple_chkpt_dir(self, fit_name=None):
        self.chkpt_dir = str(checkpoint_path)
        os.makedirs(self.chkpt_dir, exist_ok=True)
    model.create_chkpt_dir = lambda fit_name=None: simple_chkpt_dir(model, fit_name)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # Initialize trainer
    trainer = TrainerVAE(
        model=model,
        cfg=train_cfg,
        device=device,
        verbose=True,
    )
    
    # Train
    start_time = time.time()
    trainer.train(
        epochs=epochs,
        fresh_fit=True,
        save=True
    )
    total_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {total_time/3600:.2f} hours")
    
    # Save checkpoint explicitly
    checkpoint_file = checkpoint_path / 'checkpoint.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_cfg': vars(model_cfg),
        'train_cfg': vars(train_cfg),
    }, checkpoint_file)
    print(f"✓ Checkpoint saved to {checkpoint_file}")
    
    # Save final metrics
    metrics = {
        'config': config,
        'total_time': total_time,
        'n_params': n_params,
    }
    
    # Evaluate final model
    model.eval()
    # Set temperature to 0 for deterministic evaluation
    model.update_t(0.0)
    with torch.no_grad():
        dl_vld = torch.utils.data.DataLoader(
            vld_data, batch_size=1000, shuffle=False, drop_last=False
        )
        x_val = next(iter(dl_vld))[0].to(device)
        dist, log_dr, spks, y = model(x_val)
        
        kl_diag = model.loss_kl(log_dr)
        mse = torch.mean((x_val - y)**2, dim=[1,2,3])
        kl_per_sample = torch.sum(kl_diag, dim=1)
        recon_loss = torch.mean(mse) * (model.cfg.input_sz ** 2)
        kl_loss = torch.mean(kl_per_sample)
        neg_elbo = recon_loss + kl_loss
        
        # Active neurons
        kl_per_dim = torch.mean(kl_diag, dim=0).cpu().numpy()
        n_active = np.sum(kl_per_dim > 0.01)
        
        # Lifetime sparsity
        spks_np = spks.cpu().numpy()
        active_per_neuron = np.mean(spks_np > 0, axis=0)
        lifetime_sparsity = np.mean(active_per_neuron)
        
        metrics['neg_elbo'] = neg_elbo.item()
        metrics['recon_loss'] = recon_loss.item()
        metrics['kl_loss'] = kl_loss.item()
        metrics['n_active'] = int(n_active)
        metrics['lifetime_sparsity'] = float(lifetime_sparsity)
        metrics['kl_per_dim'] = kl_per_dim.tolist()
    
    # Save metrics
    metrics_file = checkpoint_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save sweep config separately
    config_file = checkpoint_path / 'sweep_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Metrics saved to {metrics_file}")
    print(f"  Negative ELBO: {metrics['neg_elbo']:.2f}")
    print(f"  Active neurons: {metrics['n_active']}/512")
    print(f"  Lifetime sparsity: {metrics['lifetime_sparsity']:.4f}")
    
    return metrics


def main():
    args = parse_args()
    
    # Parse config
    if os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)
    
    # Train
    metrics = train_model(config, args.gpu)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
