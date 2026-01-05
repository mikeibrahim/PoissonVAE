#!/usr/bin/env python3
"""
GPU Job Queue Manager for parameter sweeps.
Runs jobs across multiple GPUs asynchronously.

Usage: python scripts/gpu_sweep_manager.py --gpu GPU_ID
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from filelock import FileLock

# Get the project root
PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_PATH = PROJECT_ROOT / '.venv' / 'bin' / 'python'

# Job queue file
QUEUE_FILE = Path('./checkpoints/sweep/job_queue.json')
LOCK_FILE = Path('./checkpoints/sweep/job_queue.lock')
COMPLETED_FILE = Path('./checkpoints/sweep/completed_jobs.json')


def init_queue():
    """Initialize the job queue with all configurations."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate all configurations
    configs = []
    
    # Parameters
    temps = [0.01, 0.1, 0.4]
    anneals = [True, False]
    seeds = list(range(3))
    epochs = 2000
    
    # Poisson configurations
    n_exps = [16, 64, 256]
    for temp in temps:
        for anneal in anneals:
            for n_exp in n_exps:
                for seed in seeds:
                    config = {
                        'dist_type': 'poisson',
                        'temp': temp,
                        'anneal': anneal,
                        'n_param': n_exp,
                        'seed': seed,
                        'epochs': epochs,
                        'status': 'pending',
                    }
                    configs.append(config)
    
    # GumbelSoftmaxPoisson configurations
    upperbounds = [5, 10, 15]
    for temp in temps:
        for anneal in anneals:
            for ub in upperbounds:
                for seed in seeds:
                    config = {
                        'dist_type': 'gumbel',
                        'temp': temp,
                        'anneal': anneal,
                        'n_param': ub,
                        'seed': seed,
                        'epochs': epochs,
                        'status': 'pending',
                    }
                    configs.append(config)
    
    # Save queue
    with open(QUEUE_FILE, 'w') as f:
        json.dump({'jobs': configs}, f, indent=2)
    
    # Initialize completed jobs file
    with open(COMPLETED_FILE, 'w') as f:
        json.dump({'completed': []}, f, indent=2)
    
    print(f"Initialized job queue with {len(configs)} jobs")
    print(f"  Poisson configs: {len([c for c in configs if c['dist_type'] == 'poisson'])}")
    print(f"  Gumbel configs: {len([c for c in configs if c['dist_type'] == 'gumbel'])}")
    
    return configs


def get_next_job(gpu_id: int):
    """Get the next pending job from the queue (thread-safe)."""
    lock = FileLock(str(LOCK_FILE))
    
    with lock:
        if not QUEUE_FILE.exists():
            return None
        
        with open(QUEUE_FILE, 'r') as f:
            data = json.load(f)
        
        jobs = data['jobs']
        
        # Find first pending job
        for i, job in enumerate(jobs):
            if job['status'] == 'pending':
                # Mark as running
                jobs[i]['status'] = f'running_gpu{gpu_id}'
                jobs[i]['gpu'] = gpu_id
                jobs[i]['start_time'] = time.time()
                
                # Save updated queue
                with open(QUEUE_FILE, 'w') as f:
                    json.dump({'jobs': jobs}, f, indent=2)
                
                return job
        
        return None


def mark_job_complete(config: dict, gpu_id: int, success: bool = True):
    """Mark a job as completed."""
    lock = FileLock(str(LOCK_FILE))
    
    with lock:
        # Update queue
        with open(QUEUE_FILE, 'r') as f:
            data = json.load(f)
        
        jobs = data['jobs']
        for i, job in enumerate(jobs):
            if (job['dist_type'] == config['dist_type'] and 
                job['temp'] == config['temp'] and
                job['anneal'] == config['anneal'] and
                job['n_param'] == config['n_param'] and
                job['seed'] == config['seed']):
                jobs[i]['status'] = 'completed' if success else 'failed'
                jobs[i]['end_time'] = time.time()
                break
        
        with open(QUEUE_FILE, 'w') as f:
            json.dump({'jobs': jobs}, f, indent=2)
        
        # Update completed jobs
        with open(COMPLETED_FILE, 'r') as f:
            completed_data = json.load(f)
        
        completed_data['completed'].append({
            **config,
            'gpu': gpu_id,
            'success': success,
            'timestamp': time.time(),
        })
        
        with open(COMPLETED_FILE, 'w') as f:
            json.dump(completed_data, f, indent=2)


def get_queue_status():
    """Get current queue status."""
    if not QUEUE_FILE.exists():
        return {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0, 'total': 0}
    
    with open(QUEUE_FILE, 'r') as f:
        data = json.load(f)
    
    jobs = data['jobs']
    status = {
        'pending': len([j for j in jobs if j['status'] == 'pending']),
        'running': len([j for j in jobs if j['status'].startswith('running')]),
        'completed': len([j for j in jobs if j['status'] == 'completed']),
        'failed': len([j for j in jobs if j['status'] == 'failed']),
        'total': len(jobs),
    }
    return status


def run_job(config: dict, gpu_id: int):
    """Run a single training job."""
    config_str = json.dumps(config)
    
    cmd = [
        str(PYTHON_PATH),
        'scripts/train_sweep.py',
        '--config', config_str,
        '--gpu', str(gpu_id),
    ]
    
    model_name = f"{config['dist_type']}_temp{config['temp']}_anneal{config['anneal']}_n{config['n_param']}_seed{config['seed']}"
    log_file = Path(f'./checkpoints/sweep/{model_name}/training.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[GPU {gpu_id}] Starting: {model_name}")
    
    with open(log_file, 'w') as f:
        process = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    
    success = process.returncode == 0
    if success:
        print(f"[GPU {gpu_id}] ✓ Completed: {model_name}")
    else:
        print(f"[GPU {gpu_id}] ✗ Failed: {model_name}")
    
    return success


def gpu_worker(gpu_id: int):
    """Worker loop for a single GPU."""
    print(f"\n{'='*60}")
    print(f"GPU {gpu_id} Worker Started")
    print(f"{'='*60}")
    
    jobs_completed = 0
    
    while True:
        # Get next job
        job = get_next_job(gpu_id)
        
        if job is None:
            print(f"\n[GPU {gpu_id}] No more jobs in queue. Worker finished.")
            break
        
        # Run the job
        try:
            success = run_job(job, gpu_id)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            success = False
        
        # Mark complete
        mark_job_complete(job, gpu_id, success)
        jobs_completed += 1
        
        # Print status
        status = get_queue_status()
        print(f"\n[GPU {gpu_id}] Queue status: {status['completed']}/{status['total']} completed, "
              f"{status['pending']} pending, {status['running']} running")
    
    print(f"\n[GPU {gpu_id}] Total jobs completed: {jobs_completed}")


def parse_args():
    parser = argparse.ArgumentParser(description='GPU Sweep Manager')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')
    parser.add_argument('--init', action='store_true', help='Initialize job queue')
    parser.add_argument('--worker', action='store_true', help='Run as worker')
    parser.add_argument('--status', action='store_true', help='Show queue status')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.init:
        init_queue()
        return
    
    if args.status:
        status = get_queue_status()
        print(f"Queue Status:")
        print(f"  Pending:   {status['pending']}")
        print(f"  Running:   {status['running']}")
        print(f"  Completed: {status['completed']}")
        print(f"  Failed:    {status['failed']}")
        print(f"  Total:     {status['total']}")
        return
    
    if args.worker:
        if args.gpu is None:
            print("Error: --gpu required with --worker")
            sys.exit(1)
        gpu_worker(args.gpu)
        return
    
    print("Usage:")
    print("  --init          Initialize job queue")
    print("  --worker --gpu N  Start worker on GPU N")
    print("  --status        Show queue status")


if __name__ == '__main__':
    main()
