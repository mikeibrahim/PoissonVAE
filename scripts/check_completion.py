#!/usr/bin/env python3
"""Check completion status of all Poisson sweep runs."""

import json
from pathlib import Path

def check_poisson_completion(checkpoint_dir='./checkpoints/sweep'):
    """Check if all Poisson runs completed 3000 epochs."""
    checkpoint_dir = Path(checkpoint_dir)
    
    incomplete_runs = []
    complete_runs = []
    
    for run_dir in sorted(checkpoint_dir.glob('poisson_*')):
        if not run_dir.is_dir():
            continue
        
        # Find the highest epoch checkpoint
        checkpoints = list(run_dir.glob('PoissonVAE+TrainerVAE-*.pt'))
        
        if not checkpoints:
            print(f"❌ No checkpoints found: {run_dir.name}")
            incomplete_runs.append((run_dir.name, 0))
            continue
        
        # Extract epoch numbers
        epochs = []
        for ckpt in checkpoints:
            # Format: PoissonVAE+TrainerVAE-NNNN_(date).pt
            name = ckpt.stem
            parts = name.split('-')
            if len(parts) >= 2:
                epoch_str = parts[1].split('_')[0]
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    pass
        
        if epochs:
            max_epoch = max(epochs)
            if max_epoch < 3000:
                print(f"⚠️  Incomplete: {run_dir.name} (epoch {max_epoch}/3000)")
                incomplete_runs.append((run_dir.name, max_epoch))
            else:
                print(f"✅ Complete: {run_dir.name} (epoch {max_epoch})")
                complete_runs.append((run_dir.name, max_epoch))
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Complete: {len(complete_runs)}/36")
    print(f"  Incomplete: {len(incomplete_runs)}/36")
    print(f"{'='*70}")
    
    if incomplete_runs:
        print(f"\nIncomplete runs:")
        for run_name, epoch in incomplete_runs:
            print(f"  • {run_name}: epoch {epoch}")
    
    return incomplete_runs, complete_runs

if __name__ == '__main__':
    check_poisson_completion()
