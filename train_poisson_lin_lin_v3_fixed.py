"""
Train poisson_lin_lin_v3 with proper checkpoint saving.
This version allows n_exp to be updated dynamically during training (unlike the sweep models).
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, './')

from main.config_vae import ConfigPoisVAE, ConfigTrainVAE
from main.vae import PoissonVAE
from main.train_vae import TrainerVAE
from base.dataset import make_dataset

def train_poisson_lin_lin_v3():
    """Train poisson_lin_lin_v3 with proper n_exp handling."""
    
    # Configuration
    DATASET = "vH16"
    DEVICE_IDX = 0
    DEVICE = f"cuda:{DEVICE_IDX}" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints"
    MODEL_NAME = "poisson_lin_lin_v3"
    EPOCHS = 1500
    BATCH_SIZE = 1000
    
    print("=" * 70)
    print("TRAINING: poisson_lin_lin_v3")
    print("=" * 70)
    
    # Load base configs from original poisson_lin_lin
    with open('./checkpoints/poisson_lin_lin/ConfigPoisVAE.json', 'r') as f:
        base_model_config = json.load(f)
    
    with open('./checkpoints/poisson_lin_lin/ConfigTrainVAE.json', 'r') as f:
        base_train_config = json.load(f)
    
    # Modify for v3
    base_train_config['epochs'] = EPOCHS
    base_train_config['scheduler_kws']['T_max'] = float(EPOCHS - 5)  # Adjust for warmup
    base_train_config['chkpt_freq'] = EPOCHS  # Save only final checkpoint
    
    # Setup checkpoint directory
    checkpoint_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save configs
    with open(os.path.join(checkpoint_path, 'ConfigPoisVAE.json'), 'w') as f:
        json.dump(base_model_config, f, indent=2)
    
    with open(os.path.join(checkpoint_path, 'ConfigTrainVAE.json'), 'w') as f:
        json.dump(base_train_config, f, indent=2)
    
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
    
    # Create device
    device = torch.device(DEVICE)
    print(f"\nDevice: {device}")
    
    # Load dataset
    print(f"Loading {DATASET} dataset...")
    trn_data, vld_data, tst_data = make_dataset(
        dataset=DATASET,
        device=device,
        load_dir="Datasets",
    )
    print(f"  Training samples: {len(trn_data)}")
    print(f"  Validation samples: {len(vld_data)}")
    
    # Create model
    print("\nCreating model...")
    model = PoissonVAE(model_cfg)
    model = model.to(device)
    print(f"  Model: {model_cfg.enc_type}|{model_cfg.dec_type}")
    print(f"  Latents: {model_cfg.n_latents}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = TrainerVAE(
        model=model,
        cfg=train_cfg,
        device=str(device),
        verbose=True,
    )
    
    # Create checkpoint dir override
    def create_chkpt_dir(self, fit_name=None):
        self.chkpt_dir = checkpoint_path
        os.makedirs(self.chkpt_dir, exist_ok=True)
    
    model.create_chkpt_dir = lambda fit_name=None: create_chkpt_dir(model, fit_name)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.lr}")
    print(f"  Temperature: {train_cfg.temp_start} → {train_cfg.temp_stop}")
    print(f"  Temp anneal portion: {train_cfg.temp_anneal_portion}")
    print(f"  Checkpoint freq: {train_cfg.chkpt_freq}")
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    trainer.train(fit_name=MODEL_NAME, save=True)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Checkpoint directory: {checkpoint_path}")
    print("=" * 70)
    
    # List checkpoint files
    ckpt_files = list(Path(checkpoint_path).glob("*.pt"))
    if ckpt_files:
        print(f"\nCheckpoint files saved:")
        for f in sorted(ckpt_files):
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("\n⚠ No checkpoint files found!")
    
    return trainer, model, checkpoint_path

if __name__ == '__main__':
    trainer, model, checkpoint_path = train_poisson_lin_lin_v3()
