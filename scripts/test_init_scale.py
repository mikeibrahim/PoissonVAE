#!/usr/bin/env python3
"""
Test training with init_scale=0.05 (matching original poisson_lin_lin).
This is a single model test to verify the fix before running the full sweep.
"""

import os
import sys
import json
import time
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from main.vae import PoissonVAE
from main.train_vae import TrainerVAE
from main.config_vae import ConfigPoisVAE, ConfigTrainVAE
from base.dataset import make_dataset


def main():
    gpu = 0
    device = torch.device(f'cuda:{gpu}')
    
    # Test config - same as a sweep_2 config but with init_scale=0.05
    checkpoint_path = Path('./checkpoints/test_init_scale_fix')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Model config with CORRECTED init_scale=0.05 (matching original)
    model_config = {
        "prior_clamp": -4,
        "prior_log_dist": "uniform",
        "hard_fwd": False,
        "exc_only": False,
        "rmax_q": 1.0,
        "dataset": "vH16",
        "n_ch": 32,
        "n_latents": 512,
        "enc_type": "lin",
        "dec_type": "lin",
        "enc_bias": False,
        "dec_bias": False,
        "enc_norm": False,
        "dec_norm": False,
        "fit_prior": True,
        "activation_fn": "swish",
        "init_dist": "Normal",
        "init_scale": 0.05,  # KEY FIX: was 0.0001 in sweep, should be 0.05
        "res_eps": 1.0,
        "use_bn": False,
        "use_se": True,
        "seed": 0,
    }
    
    # Training config with temperature annealing (same as original)
    train_config = {
        "method": "mc",
        "kl_beta": 1.0,
        "kl_beta_min": 0.0001,
        "kl_anneal_cycles": 0,
        "kl_anneal_portion": 0.5,
        "kl_const_portion": 0.0,
        "lambda_anneal": False,
        "lambda_init": 0.0,
        "lambda_norm": 0.0,
        "temp_anneal_portion": 0.5,
        "temp_anneal_type": "lin",
        "temp_start": 1.0,
        "temp_stop": 0.05,
        "lr": 0.005,
        "epochs": 1500,  # Longer test run (1500 epochs instead of 500)
        "batch_size": 1000,
        "warm_restart": 0,
        "warmup_epochs": 5,
        "optimizer": "adamax_fast",
        "optimizer_kws": {
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-08
        },
        "scheduler_type": "cosine",
        "scheduler_kws": {
            "T_max": 1495.0,
            "eta_min": 1e-05
        },
        "ema_rate": None,
        "grad_clip": 500,
        "use_amp": False,
        "chkpt_freq": 1500,
        "eval_freq": 20,
        "log_freq": 10
    }
    
    print("="*70)
    print("TEST: Training with init_scale=0.05 (matching original poisson_lin_lin)")
    print("="*70)
    print(f"\nKey difference from sweep_2:")
    print(f"  init_scale: 0.05 (was 0.0001 in sweep)")
    print(f"  epochs: 500 (short test run)")
    print(f"\nExpected outcome:")
    print(f"  - n_exp should increase to ~60-70 (was ~17-20 with wrong init_scale)")
    print(f"  - Decoder weights should have range ~[-0.2, 0.4] (was [-0.08, 0.08])")
    print(f"  - Dictionary should show Gabor-like edge filters (not noise)")
    
    # Save configs
    with open(checkpoint_path / 'ConfigPoisVAE.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    with open(checkpoint_path / 'ConfigTrainVAE.json', 'w') as f:
        json.dump(train_config, f, indent=2)
    
    # Create model config object
    model_cfg = ConfigPoisVAE(dataset=model_config['dataset'])
    for k, v in model_config.items():
        if hasattr(model_cfg, k) and k != 'dataset':
            setattr(model_cfg, k, v)
    
    # Create training config object
    train_cfg = ConfigTrainVAE()
    for k, v in train_config.items():
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
    
    print(f"  init_scale: {model_cfg.init_scale}")
    print(f"  Initial n_exp: {model.n_exp.item()}")
    
    # Check initial weight scale
    init_weights = model.fc_dec.get_weight().detach().cpu().numpy()
    print(f"  Initial decoder weight range: [{init_weights.min():.4f}, {init_weights.max():.4f}]")
    
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
    print("Starting training (1500 epochs)...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    trainer.train(fit_name='test_init_scale', save=True)
    train_time = time.time() - start_time
    
    # Final stats
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Training time: {train_time/60:.1f} minutes")
    print(f"  Final n_exp: {model.n_exp.item()}")
    
    final_weights = model.fc_dec.get_weight().detach().cpu().numpy()
    print(f"  Final decoder weight range: [{final_weights.min():.4f}, {final_weights.max():.4f}]")
    
    # Compare with expected values
    print(f"\n  Comparison:")
    print(f"    Original poisson_lin_lin: n_exp=66, weights=[-0.20, 0.39]")
    print(f"    Sweep_2 (wrong init):     n_exp=17, weights=[-0.08, 0.08]")
    print(f"    This test:                n_exp={model.n_exp.item()}, weights=[{final_weights.min():.2f}, {final_weights.max():.2f}]")
    
    # Save metrics
    metrics = {
        'final_n_exp': int(model.n_exp.item()),
        'final_weight_min': float(final_weights.min()),
        'final_weight_max': float(final_weights.max()),
        'train_time': train_time,
        'init_scale': model_cfg.init_scale,
    }
    with open(checkpoint_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if model.n_exp.item() > 40:
        print(f"\n✓ SUCCESS: n_exp > 40, init_scale fix is working!")
    else:
        print(f"\n✗ FAILED: n_exp still low, need to investigate further")


if __name__ == '__main__':
    main()
