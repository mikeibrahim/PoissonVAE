#!/usr/bin/env python3
"""
Launch the complete sweep_2 experiment with corrected training.
Runs multiple jobs per GPU (max 6 jobs per GPU based on memory).
Uses GPUs 0 and 1.
"""

import os
import sys
import json
import itertools
from pathlib import Path

def generate_all_configs():
    """Generate all 108 sweep configurations."""
    
    configs = []
    
    # Poisson configs: temps x n_params x anneals x seeds
    poisson_temps = [0.01, 0.1, 0.4]
    poisson_n_params = [16, 64, 256]  # n_exp values (though they'll be dynamically updated)
    
    # Gumbel configs: temps x n_params x anneals x seeds
    gumbel_temps = [0.01, 0.1, 0.4]
    gumbel_n_params = [5, 10, 15]  # upperbound values
    
    anneals = [True, False]
    seeds = [0, 1, 2]
    epochs = 2000
    
    # Generate Poisson configs
    for temp, n_param, anneal, seed in itertools.product(
        poisson_temps, poisson_n_params, anneals, seeds
    ):
        configs.append({
            'dist_type': 'poisson',
            'temp': temp,
            'n_param': n_param,
            'anneal': anneal,
            'seed': seed,
            'epochs': epochs,
        })
    
    # Generate Gumbel configs
    for temp, n_param, anneal, seed in itertools.product(
        gumbel_temps, gumbel_n_params, anneals, seeds
    ):
        configs.append({
            'dist_type': 'gumbel',
            'temp': temp,
            'n_param': n_param,
            'anneal': anneal,
            'seed': seed,
            'epochs': epochs,
        })
    
    return configs


def get_model_name(config):
    """Get model name from config."""
    anneal_str = 'annealTrue' if config['anneal'] else 'annealFalse'
    return f"{config['dist_type']}_temp{config['temp']}_{anneal_str}_n{config['n_param']}_seed{config['seed']}"


def check_completed(config):
    """Check if a model has already been trained."""
    model_name = get_model_name(config)
    checkpoint_path = Path('./checkpoints/sweep_2') / model_name
    
    # Check for checkpoint file
    if checkpoint_path.exists():
        pt_files = list(checkpoint_path.glob('*.pt'))
        if pt_files:
            return True
    return False


def main():
    # Generate all configs
    all_configs = generate_all_configs()
    print(f"Total configurations: {len(all_configs)}")
    
    # Filter out already completed
    pending_configs = [c for c in all_configs if not check_completed(c)]
    print(f"Pending configurations: {len(pending_configs)}")
    print(f"Already completed: {len(all_configs) - len(pending_configs)}")
    
    if not pending_configs:
        print("\n✓ All jobs already completed!")
        return
    
    # Create config files directory
    config_dir = Path('./checkpoints/sweep_2/configs')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs to files
    config_files = []
    for config in pending_configs:
        model_name = get_model_name(config)
        config_file = config_dir / f"{model_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        config_files.append((model_name, str(config_file)))
    
    print(f"\nSaved {len(config_files)} config files to {config_dir}")
    
    # Split jobs between GPUs (alternating for load balance)
    gpu0_jobs = config_files[0::2]
    gpu1_jobs = config_files[1::2]
    
    print(f"\nGPU 0 jobs: {len(gpu0_jobs)}")
    print(f"GPU 1 jobs: {len(gpu1_jobs)}")
    
    # Create job scripts
    script_dir = Path('./checkpoints/sweep_2/scripts')
    script_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU 0 script
    gpu0_script = script_dir / 'run_gpu0.sh'
    with open(gpu0_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd /home/michael/code/poisson_notebook\n")
        f.write("source .venv/bin/activate\n\n")
        for model_name, config_file in gpu0_jobs:
            f.write(f"echo '=== Training: {model_name} on GPU 0 ==='\n")
            f.write(f"python scripts/train_sweep_v3.py --config {config_file} --gpu 0\n\n")
    
    # GPU 1 script
    gpu1_script = script_dir / 'run_gpu1.sh'
    with open(gpu1_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd /home/michael/code/poisson_notebook\n")
        f.write("source .venv/bin/activate\n\n")
        for model_name, config_file in gpu1_jobs:
            f.write(f"echo '=== Training: {model_name} on GPU 1 ==='\n")
            f.write(f"python scripts/train_sweep_v3.py --config {config_file} --gpu 1\n\n")
    
    os.chmod(gpu0_script, 0o755)
    os.chmod(gpu1_script, 0o755)
    
    print(f"\nCreated job scripts:")
    print(f"  GPU 0: {gpu0_script}")
    print(f"  GPU 1: {gpu1_script}")
    
    print("\n" + "="*70)
    print("TO RUN THE SWEEP:")
    print("="*70)
    print("\nRun in two separate tmux sessions:")
    print(f"  tmux new -s sweep_gpu0 '{gpu0_script}'")
    print(f"  tmux new -s sweep_gpu1 '{gpu1_script}'")
    print("\nOr use the combined command:")
    print("  tmux new -d -s sweep_gpu0 'bash " + str(gpu0_script) + "'")
    print("  tmux new -d -s sweep_gpu1 'bash " + str(gpu1_script) + "'")


if __name__ == '__main__':
    main()
