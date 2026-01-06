#!/usr/bin/env python3
"""
Generate sweep_2 jobs and launch them on GPUs 0 and 1.
Runs multiple jobs concurrently on each GPU.
"""

import json
import subprocess
import time
from pathlib import Path

# Sweep parameters (same as original sweep)
DIST_TYPES = ['poisson', 'gumbel']
TEMPS = [0.01, 0.1, 0.4]
ANNEAL = [True, False]
N_PARAMS = {
    'poisson': [16, 32, 64, 128, 256],
    'gumbel': [5, 10, 15]
}
SEEDS = [0, 1, 2]
EPOCHS = 2000

def generate_jobs():
    """Generate all job configurations."""
    jobs = []
    
    for dist_type in DIST_TYPES:
        for temp in TEMPS:
            for anneal in ANNEAL:
                for n_param in N_PARAMS[dist_type]:
                    for seed in SEEDS:
                        job = {
                            'dist_type': dist_type,
                            'temp': temp,
                            'anneal': anneal,
                            'n_param': n_param,
                            'seed': seed,
                            'epochs': EPOCHS,
                        }
                        jobs.append(job)
    
    return jobs


def get_model_name(job):
    """Get model name from job config."""
    anneal_str = 'annealTrue' if job['anneal'] else 'annealFalse'
    return f"{job['dist_type']}_temp{job['temp']}_{anneal_str}_n{job['n_param']}_seed{job['seed']}"


def job_exists(job):
    """Check if job already completed."""
    model_name = get_model_name(job)
    checkpoint_dir = Path(f'./checkpoints/sweep_2/{model_name}')
    metrics_file = checkpoint_dir / 'metrics.json'
    return metrics_file.exists()


def launch_job_tmux(job, gpu_id, session_id):
    """Launch a job in a tmux session."""
    model_name = get_model_name(job)
    config_json = json.dumps(job)
    
    # Create tmux session name
    tmux_name = f"sweep2_gpu{gpu_id}_{session_id}"
    
    # Create command
    cmd = f"""
cd /home/michael/code/poisson_notebook && \
source .venv/bin/activate && \
python scripts/train_sweep_v2.py \
    --config '{config_json}' \
    --gpu {gpu_id} \
    > logs/sweep_2/{model_name}.log 2>&1
"""
    
    # Check if tmux session exists
    check_cmd = f"tmux has-session -t {tmux_name} 2>/dev/null"
    session_exists = subprocess.call(check_cmd, shell=True) == 0
    
    if session_exists:
        # Kill existing session
        subprocess.call(f"tmux kill-session -t {tmux_name}", shell=True)
        time.sleep(0.5)
    
    # Create new tmux session
    tmux_cmd = f"tmux new-session -d -s {tmux_name} '{cmd}'"
    subprocess.call(tmux_cmd, shell=True)
    
    print(f"  Launched: {model_name} on GPU {gpu_id} (session: {tmux_name})")
    
    return tmux_name


def get_active_sessions(prefix="sweep2_gpu"):
    """Get list of active tmux sessions."""
    try:
        result = subprocess.check_output("tmux list-sessions", shell=True, text=True)
        sessions = []
        for line in result.strip().split('\n'):
            if prefix in line:
                session_name = line.split(':')[0]
                sessions.append(session_name)
        return sessions
    except subprocess.CalledProcessError:
        return []


def main():
    print("=" * 70)
    print("SWEEP_2 JOB LAUNCHER")
    print("=" * 70)
    
    # Create log directory
    Path('./logs/sweep_2').mkdir(parents=True, exist_ok=True)
    
    # Generate all jobs
    all_jobs = generate_jobs()
    print(f"\nTotal jobs: {len(all_jobs)}")
    
    # Filter out completed jobs
    pending_jobs = [job for job in all_jobs if not job_exists(job)]
    completed_count = len(all_jobs) - len(pending_jobs)
    
    print(f"Completed: {completed_count}")
    print(f"Pending: {len(pending_jobs)}")
    
    if not pending_jobs:
        print("\n✓ All jobs already completed!")
        return
    
    # Configuration
    GPUS = [0, 1]
    JOBS_PER_GPU = 6  # Run 6 jobs per GPU concurrently
    
    print(f"\nConfiguration:")
    print(f"  GPUs: {GPUS}")
    print(f"  Jobs per GPU: {JOBS_PER_GPU}")
    print(f"  Total concurrent: {len(GPUS) * JOBS_PER_GPU}")
    
    # Launch jobs
    print(f"\n{'='*70}")
    print("LAUNCHING JOBS")
    print(f"{'='*70}")
    
    job_index = 0
    gpu_sessions = {gpu: [] for gpu in GPUS}
    
    # Initial launch - fill up all GPU slots
    for gpu in GPUS:
        for slot in range(JOBS_PER_GPU):
            if job_index < len(pending_jobs):
                job = pending_jobs[job_index]
                session_name = launch_job_tmux(job, gpu, slot)
                gpu_sessions[gpu].append({
                    'session': session_name,
                    'job': job,
                    'slot': slot,
                })
                job_index += 1
                time.sleep(0.2)  # Small delay between launches
    
    print(f"\nInitial launch complete: {job_index} jobs started")
    
    # Monitor and refill
    print(f"\nMonitoring and refilling slots...")
    print(f"{'='*70}")
    
    while job_index < len(pending_jobs):
        time.sleep(30)  # Check every 30 seconds
        
        active_sessions = get_active_sessions()
        
        # Check each GPU
        for gpu in GPUS:
            for slot_info in gpu_sessions[gpu]:
                session_name = slot_info['session']
                
                # If session no longer active, refill the slot
                if session_name not in active_sessions:
                    model_name = get_model_name(slot_info['job'])
                    print(f"\n  Completed: {model_name} (GPU {gpu})")
                    
                    # Launch next job in this slot
                    if job_index < len(pending_jobs):
                        job = pending_jobs[job_index]
                        new_session = launch_job_tmux(job, gpu, slot_info['slot'])
                        slot_info['session'] = new_session
                        slot_info['job'] = job
                        job_index += 1
                        
                        print(f"  Progress: {job_index}/{len(pending_jobs)} ({job_index/len(pending_jobs)*100:.1f}%)")
    
    # Wait for all remaining jobs to complete
    print(f"\n{'='*70}")
    print("All jobs launched. Waiting for completion...")
    print(f"{'='*70}\n")
    
    while True:
        active_sessions = get_active_sessions()
        
        if not active_sessions:
            break
        
        print(f"  Active sessions: {len(active_sessions)}")
        time.sleep(60)
    
    print(f"\n{'='*70}")
    print("✓ ALL JOBS COMPLETED!")
    print(f"{'='*70}")
    
    # Final summary
    completed_jobs = [job for job in all_jobs if job_exists(job)]
    print(f"\nFinal status:")
    print(f"  Total jobs: {len(all_jobs)}")
    print(f"  Completed: {len(completed_jobs)}")
    print(f"  Success rate: {len(completed_jobs)/len(all_jobs)*100:.1f}%")


if __name__ == '__main__':
    main()
