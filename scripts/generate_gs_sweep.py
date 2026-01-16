#!/usr/bin/env python3
"""Generate GumbelSoftmax sweep configurations."""

import json
import itertools
from pathlib import Path

def generate_gs_sweep(
    dataset: str = 'vH16',
    architecture: str = 'lin|lin',
    temp_stop_values: list = [0.01, 0.1, 0.3],
    temp_anneal_values: list = [True, False],
    upperbound_param_values: list = [25, 50, 100],
    seed_values: list = [0, 1, 2],
    output_file: str = 'sweep_gs_queue.jsonl'
):
    """Generate GumbelSoftmax sweep configurations."""
    jobs = []
    run_id = 0
    
    for temp_stop, temp_anneal, upperbound_param, seed in itertools.product(
        temp_stop_values,
        temp_anneal_values,
        upperbound_param_values,
        seed_values
    ):
        job = {
            'run_id': run_id,
            'model_type': 'gumbel_poisson',  # Use gumbel_poisson model type
            'dataset': dataset,
            'architecture': architecture,
            'params': {
                'temp_stop': temp_stop,
                'temp_anneal': temp_anneal,
                'upperbound_method': 'fixed',
                'upperbound_param': upperbound_param,
                'seed': seed,
            }
        }
        jobs.append(job)
        run_id += 1
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for job in jobs:
            f.write(json.dumps(job) + '\n')
    
    print(f"✅ Generated {len(jobs)} GumbelSoftmax jobs")
    print(f"   Queue file: {output_path.absolute()}")
    print(f"\nBreakdown:")
    print(f"  • temp_stop: {temp_stop_values}")
    print(f"  • temp_anneal: {temp_anneal_values}")
    print(f"  • upperbound_param: {upperbound_param_values}")
    print(f"  • seeds: {seed_values}")
    print(f"  • Total: {len(jobs)} configurations")
    
    return jobs

if __name__ == '__main__':
    generate_gs_sweep()
