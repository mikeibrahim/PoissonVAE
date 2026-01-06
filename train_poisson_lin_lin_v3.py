"""
Retrain poisson_lin_lin with 1500 epochs (poisson_lin_lin_v3)
Using the CORRECT training procedure from p-vae.ipynb (allowing n_exp to update dynamically)
"""

import json
import torch
from pathlib import Path
from main.vae import PoissonVAE
from main.config_vae import ConfigPoisVAE, ConfigTrainVAE
from main.train_vae import TrainerVAE
import os

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load base configs from poisson_lin_lin
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

base_model_config = load_config('./checkpoints/poisson_lin_lin/ConfigPoisVAE.json')
base_train_config = load_config('./checkpoints/poisson_lin_lin/ConfigTrainVAE.json')

# Create save directory
save_dir = './checkpoints/poisson_lin_lin_v3'
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Modify epochs to 1500
base_train_config['epochs'] = 1500
base_train_config['scheduler_kws']['T_max'] = 2995.0  # Keep same as original (will be multiplied by batch count)

# Save configs
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

# Create trainer - this will load dataset automatically
print("\nCreating trainer...")
trainer = TrainerVAE(
    model=model,
    cfg=train_cfg,
    device=str(device),
    verbose=True,
)

# Train with fresh_fit=True (important! allows proper scheduler initialization)
print("\nStarting training...")
print("=" * 70)

fit_name = Path(save_dir).name
trainer.train(fit_name=fit_name, save=True, fresh_fit=True)

print("\n" + "=" * 70)
print(f"Training complete! Model saved to: {trainer.model.chkpt_dir}")
print("=" * 70)
