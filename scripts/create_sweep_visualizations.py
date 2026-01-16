#!/usr/bin/env python3
"""
Create visualizations comparing Poisson sigmoid and GumbelSoftmax runs.
Generates:
1. Neg ELBO vs Temperature
2. Neg ELBO vs Annealing
3. KL vs Reconstruction Loss
4. Lifetime Sparsity vs Temperature
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re

sns.set_style("whitegrid")
sns.set_palette("husl")

def extract_run_info(run_dir):
    """Extract configuration from run directory."""
    name = run_dir.name
    
    # Determine model type
    if "sigmoid" in name:
        model_type = "Poisson (sigmoid)"
    elif "gumbel" in name:
        model_type = "GumbelSoftmax"
    elif "cubic" in name:
        model_type = "Poisson (cubic)"
    else:
        model_type = "Unknown"
    
    # Extract seed
    seed_match = re.search(r'seed(\d+)', name)
    seed = int(seed_match.group(1)) if seed_match else None
    
    # Load config
    config_file = run_dir / "ConfigTrainVAE.json"
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    # Extract parameters
    temp_stop = config.get('temp_stop', None)
    temp_anneal = config.get('temp_anneal', None)
    indicator_approx = config.get('indicator_approx', 'sigmoid')
    
    # Also try to extract from name
    if temp_stop is None:
        temp_match = re.search(r'temp([\d\.]+)', name.replace('_', ''))
        if temp_match:
            temp_stop = float(temp_match.group(1))
    
    if temp_anneal is None:
        temp_anneal = 'anneal' in name and 'noanneal' not in name
    
    return {
        'model_type': model_type,
        'seed': seed,
        'temp_stop': temp_stop,
        'temp_anneal': temp_anneal,
        'indicator_approx': indicator_approx,
        'config': config,
        'path': run_dir
    }

def load_checkpoint_stats(run_dir):
    """Load final checkpoint and extract stats."""
    # Find the highest epoch checkpoint
    checkpoints = list(run_dir.glob("*VAE+TrainerVAE-*.pt"))
    if not checkpoints:
        return None
    
    # Get the last checkpoint
    epochs = []
    for cp in checkpoints:
        match = re.search(r'VAE-(\d+)_', cp.name)
        if match:
            epochs.append((int(match.group(1)), cp))
    
    if not epochs:
        return None
    
    _, checkpoint_file = max(epochs, key=lambda x: x[0])
    
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        # Extract stats
        stats = {}
        
        # Try to get ELBO components
        if 'trainer_stats' in checkpoint:
            trainer_stats = checkpoint['trainer_stats']
            # Get the last values
            if 'nll' in trainer_stats:
                stats['recon_loss'] = np.mean(list(trainer_stats['nll'].values())[-10:])
            if 'kl' in trainer_stats:
                stats['kl'] = np.mean(list(trainer_stats['kl'].values())[-10:])
            if 'elbo' in trainer_stats:
                stats['neg_elbo'] = -np.mean(list(trainer_stats['elbo'].values())[-10:])
            elif 'nll' in trainer_stats and 'kl' in trainer_stats:
                stats['neg_elbo'] = stats['recon_loss'] + stats['kl']
        
        # Try to get sparsity from model state
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
            if 'log_prior' in model_state:
                log_prior = model_state['log_prior']
                prior = torch.exp(log_prior)
                # Lifetime sparsity: fraction of neurons that are typically inactive
                stats['lifetime_sparsity'] = float(torch.mean((prior < 0.1).float()))
        
        return stats
    except Exception as e:
        print(f"Error loading {checkpoint_file}: {e}")
        return None

def collect_data(sweep_dir):
    """Collect all run data."""
    data = []
    
    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        info = extract_run_info(run_dir)
        stats = load_checkpoint_stats(run_dir)
        
        if stats:
            data.append({**info, **stats})
    
    return data

def plot_neg_elbo_vs_temp(data, save_path):
    """Plot negative ELBO vs temperature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by model type
    for model_type in sorted(set(d['model_type'] for d in data)):
        model_data = [d for d in data if d['model_type'] == model_type]
        
        # Group by temp
        temp_groups = defaultdict(list)
        for d in model_data:
            if d.get('temp_stop') and d.get('neg_elbo'):
                temp_groups[d['temp_stop']].append(d['neg_elbo'])
        
        temps = sorted(temp_groups.keys())
        means = [np.mean(temp_groups[t]) for t in temps]
        stds = [np.std(temp_groups[t]) for t in temps]
        
        ax.errorbar(temps, means, yerr=stds, marker='o', capsize=5, 
                    label=model_type, linewidth=2, markersize=8)
    
    ax.set_xlabel('Temperature (temp_stop)', fontsize=12)
    ax.set_ylabel('Negative ELBO', fontsize=12)
    ax.set_title('Negative ELBO vs Temperature', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_neg_elbo_vs_anneal(data, save_path):
    """Plot negative ELBO vs annealing."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by model type and annealing
    for model_type in sorted(set(d['model_type'] for d in data)):
        model_data = [d for d in data if d['model_type'] == model_type]
        
        anneal_groups = {'Annealing': [], 'No Annealing': []}
        for d in model_data:
            if d.get('neg_elbo') is not None:
                key = 'Annealing' if d.get('temp_anneal') else 'No Annealing'
                anneal_groups[key].append(d['neg_elbo'])
        
        x_pos = [0, 1]
        means = [np.mean(anneal_groups['Annealing']) if anneal_groups['Annealing'] else 0,
                 np.mean(anneal_groups['No Annealing']) if anneal_groups['No Annealing'] else 0]
        stds = [np.std(anneal_groups['Annealing']) if anneal_groups['Annealing'] else 0,
                np.std(anneal_groups['No Annealing']) if anneal_groups['No Annealing'] else 0]
        
        offset = {'Poisson (sigmoid)': -0.2, 'GumbelSoftmax': 0, 'Poisson (cubic)': 0.2}
        x_offset = offset.get(model_type, 0)
        
        ax.bar([x + x_offset for x in x_pos], means, yerr=stds, width=0.2,
               label=model_type, capsize=5, alpha=0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Annealing', 'No Annealing'], fontsize=12)
    ax.set_ylabel('Negative ELBO', fontsize=12)
    ax.set_title('Negative ELBO: Annealing vs No Annealing', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_kl_vs_recon(data, save_path):
    """Plot KL divergence vs reconstruction loss."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_type in sorted(set(d['model_type'] for d in data)):
        model_data = [d for d in data 
                      if d['model_type'] == model_type 
                      and d.get('kl') and d.get('recon_loss')]
        
        kl_vals = [d['kl'] for d in model_data]
        recon_vals = [d['recon_loss'] for d in model_data]
        
        ax.scatter(recon_vals, kl_vals, label=model_type, alpha=0.6, s=50)
    
    ax.set_xlabel('Reconstruction Loss', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('KL Divergence vs Reconstruction Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_sparsity_vs_temp(data, save_path):
    """Plot lifetime sparsity vs temperature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_type in sorted(set(d['model_type'] for d in data)):
        model_data = [d for d in data if d['model_type'] == model_type]
        
        # Group by temp
        temp_groups = defaultdict(list)
        for d in model_data:
            if d.get('temp_stop') and d.get('lifetime_sparsity') is not None:
                temp_groups[d['temp_stop']].append(d['lifetime_sparsity'])
        
        temps = sorted(temp_groups.keys())
        means = [np.mean(temp_groups[t]) for t in temps]
        stds = [np.std(temp_groups[t]) for t in temps]
        
        ax.errorbar(temps, means, yerr=stds, marker='o', capsize=5,
                    label=model_type, linewidth=2, markersize=8)
    
    ax.set_xlabel('Temperature (temp_stop)', fontsize=12)
    ax.set_ylabel('Lifetime Sparsity', fontsize=12)
    ax.set_title('Lifetime Sparsity vs Temperature', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def main():
    sweep_dir = Path("checkpoints/sweep")
    figures_dir = Path("figures/sweep_results")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CREATING SWEEP VISUALIZATIONS")
    print("="*70)
    
    # Collect data
    print("\nCollecting data from all runs...")
    data = collect_data(sweep_dir)
    print(f"Loaded {len(data)} runs with stats")
    
    # Create plots
    print("\nGenerating visualizations...")
    plot_neg_elbo_vs_temp(data, figures_dir / "neg_elbo_vs_temp.png")
    plot_neg_elbo_vs_anneal(data, figures_dir / "neg_elbo_vs_anneal.png")
    plot_kl_vs_recon(data, figures_dir / "kl_vs_recon.png")
    plot_sparsity_vs_temp(data, figures_dir / "sparsity_vs_temp.png")
    
    print(f"\n✅ All visualizations saved to: {figures_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
