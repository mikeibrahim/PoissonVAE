#!/usr/bin/env python3
"""
Generate and run adaptive upperbound sweep experiments in tmux.
Distributes 18 experiments across available GPUs in parallel.
"""

import os
import subprocess
import itertools

# Parameters
ROOT_DIR = '/home/michael/code/wandb_dataset'
PYTHON = f'{ROOT_DIR}/.venv/bin/python'
AVAILABLE_GPUS = [0, 1, 2]  # GPU 3 is busy

# Sweep parameters
TEMPS = [0.01, 0.1, 0.3]
TEMP_ANNEALS = [True, False]
SEEDS = [0, 1, 2]

# Generate all combinations
experiments = list(itertools.product(TEMPS, TEMP_ANNEALS, SEEDS))
print(f"Total experiments: {len(experiments)}")

# Kill any existing session
SESSION_NAME = "adaptive_sweep"
subprocess.run(f"tmux kill-session -t {SESSION_NAME} 2>/dev/null", shell=True)

# Assign experiments to GPUs
gpu_assignments = {gpu: [] for gpu in AVAILABLE_GPUS}
for i, exp in enumerate(experiments):
    gpu = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
    gpu_assignments[gpu].append(exp)

# Create bash script for each GPU
scripts_dir = f'{ROOT_DIR}/scripts/tmp_sweep'
os.makedirs(scripts_dir, exist_ok=True)

for gpu, exps in gpu_assignments.items():
    if not exps:
        continue
    
    script_lines = ['#!/bin/bash', f'cd {ROOT_DIR}', '']
    
    for temp, anneal, seed in exps:
        # Format names
        if temp == 0.01:
            t_str = '001'
        elif temp == 0.1:
            t_str = '01'  
        elif temp == 0.3:
            t_str = '03'
        else:
            t_str = str(temp).replace('.', '')
        ann_str = 'ann' if anneal else 'noann'
        
        exp_name = f"gs_adaptive_t{t_str}_{ann_str}_seed{seed}"
        
        cmd = f'{PYTHON} -m main.train_vae {gpu} vH16 gumbel_poisson "lin|lin"'
        cmd += f' --seed {seed}'
        cmd += f' --temp_stop {temp}'
        cmd += f' --upperbound_method adaptive'
        cmd += f' --upperbound_param 50'
        if anneal:
            cmd += f' --temp_anneal'
        # Enable wandb logging (removed --no_wandb)
        cmd += f' 2>&1 | tee -a checkpoints/sweep/{exp_name}/TrainerVAE.log'
        
        script_lines.append(f'echo "=== Running {exp_name} on GPU {gpu} ==="')
        script_lines.append(cmd)
        script_lines.append('')
        
        print(f"GPU {gpu}: {exp_name}")
    
    # Write script
    script_path = f'{scripts_dir}/gpu{gpu}.sh'
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    os.chmod(script_path, 0o755)

# Create tmux session and windows
subprocess.run(f"tmux new-session -d -s {SESSION_NAME}", shell=True)

for i, gpu in enumerate(AVAILABLE_GPUS):
    script_path = f'{scripts_dir}/gpu{gpu}.sh'
    window_name = f"gpu{gpu}"
    
    if i == 0:
        # First window is created with the session, just rename it
        subprocess.run(f"tmux rename-window -t {SESSION_NAME}:0 {window_name}", shell=True)
    else:
        subprocess.run(f"tmux new-window -t {SESSION_NAME} -n {window_name}", shell=True)
    
    # Execute the script
    subprocess.run(f"tmux send-keys -t {SESSION_NAME}:{window_name} 'bash {script_path}' Enter", shell=True)

print(f"\nTmux session '{SESSION_NAME}' created with {len(AVAILABLE_GPUS)} windows.")
print(f"Attach with: tmux attach -t {SESSION_NAME}")
print(f"List windows: tmux list-windows -t {SESSION_NAME}")
