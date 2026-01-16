#!/usr/bin/env python3
"""
Organize sweep model checkpoints into a structured directory.
Migrates models from ~/Projects/PoissonVAE/models to ./checkpoints/sweep/
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

def find_sweep_runs(base_dir: Path) -> List[Path]:
    """Find all sweep run directories."""
    sweep_runs = []
    for model_dir in base_dir.glob('*'):
        if model_dir.is_dir():
            for run_dir in model_dir.glob('sweep_*'):
                if run_dir.is_dir():
                    sweep_runs.append(run_dir)
    return sweep_runs

def extract_run_info(run_dir: Path) -> Dict:
    """Extract run information from directory name and config."""
    dir_name = run_dir.name
    
    # Parse directory name
    parts = dir_name.split('_')
    model_type = parts[1]  # poisson, gumbel, etc.
    
    # Extract run ID and seed from directory name
    run_id = None
    seed = None
    for part in parts:
        if part.startswith('run'):
            run_id = part.replace('run', '')
        if part.startswith('seed'):
            seed = part.replace('seed', '')
    
    # Load config if available
    config_file = run_dir / 'ConfigTrainVAE.json'
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    
    return {
        'run_id': run_id,
        'model_type': model_type,
        'seed': seed,
        'dir_name': dir_name,
        'config': config,
        'source_path': run_dir
    }

def organize_checkpoints(
    source_base: str = '/home/michael/Projects/PoissonVAE/models',
    target_base: str = './checkpoints/sweep',
    copy_mode: bool = True
):
    """
    Organize sweep checkpoints into structured directory.
    
    Args:
        source_base: Base directory containing models
        target_base: Target directory for organized checkpoints
        copy_mode: If True, copy files. If False, move files.
    """
    source_base = Path(source_base)
    target_base = Path(target_base)
    target_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching for sweep runs in: {source_base}")
    sweep_runs = find_sweep_runs(source_base)
    print(f"Found {len(sweep_runs)} sweep runs\n")
    
    # Group by unique runs (handle duplicates)
    run_groups = {}
    for run_dir in sweep_runs:
        info = extract_run_info(run_dir)
        key = (info['model_type'], info['run_id'], info['seed'])
        if key not in run_groups:
            run_groups[key] = []
        run_groups[key].append((run_dir, info))
    
    print(f"Unique runs (after deduplication): {len(run_groups)}\n")
    
    # Process each unique run
    for (model_type, run_id, seed), runs in sorted(run_groups.items()):
        # Use the most recent run if there are duplicates
        runs_sorted = sorted(runs, key=lambda x: x[0].name, reverse=True)
        run_dir, info = runs_sorted[0]
        
        if len(runs) > 1:
            print(f"⚠️  Found {len(runs)} duplicates for {model_type} run {run_id} seed {seed}, using most recent")
        
        # Create target directory name
        config = info['config']
        temp_stop = config.get('temp_stop', 'NA')
        temp_anneal_portion = config.get('temp_anneal_portion', 'NA')
        indicator_approx = config.get('indicator_approx', 'NA')
        
        # Determine if temp annealing was enabled
        temp_anneal = 'anneal' if temp_anneal_portion > 0 else 'noanneal'
        
        target_dir_name = f"{model_type}_seed{seed}_run{run_id}_temp{temp_stop}_{temp_anneal}_{indicator_approx}"
        target_dir = target_base / target_dir_name
        
        # Copy/move the directory
        if target_dir.exists():
            print(f"⏭️  Skipping (already exists): {target_dir_name}")
            continue
        
        print(f"📦 {'Copying' if copy_mode else 'Moving'}: {run_dir.name}")
        print(f"   → {target_dir_name}")
        
        if copy_mode:
            shutil.copytree(run_dir, target_dir)
        else:
            shutil.move(str(run_dir), str(target_dir))
    
    print(f"\n✅ Completed! Organized {len(run_groups)} runs into: {target_base}")
    
    # Print summary
    model_types = {}
    for key in run_groups.keys():
        model_type = key[0]
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    print("\nSummary:")
    for model_type, count in sorted(model_types.items()):
        print(f"  • {model_type}: {count} runs")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize sweep checkpoints')
    parser.add_argument('--source', default='/home/michael/Projects/PoissonVAE/models',
                       help='Source directory containing models')
    parser.add_argument('--target', default='./checkpoints/sweep',
                       help='Target directory for organized checkpoints')
    parser.add_argument('--move', action='store_true',
                       help='Move files instead of copying')
    
    args = parser.parse_args()
    
    organize_checkpoints(
        source_base=args.source,
        target_base=args.target,
        copy_mode=not args.move
    )
