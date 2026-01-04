#!/usr/bin/env python3
"""
Monitor sweep progress across all GPUs.
Usage: python scripts/monitor_sweep.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

QUEUE_FILE = Path('./checkpoints/sweep/job_queue.json')
COMPLETED_FILE = Path('./checkpoints/sweep/completed_jobs.json')


def get_status():
    """Get current queue status."""
    if not QUEUE_FILE.exists():
        return None
    
    with open(QUEUE_FILE, 'r') as f:
        data = json.load(f)
    
    jobs = data['jobs']
    
    status = {
        'pending': [],
        'running': [],
        'completed': [],
        'failed': [],
    }
    
    for job in jobs:
        if job['status'] == 'pending':
            status['pending'].append(job)
        elif job['status'].startswith('running'):
            status['running'].append(job)
        elif job['status'] == 'completed':
            status['completed'].append(job)
        elif job['status'] == 'failed':
            status['failed'].append(job)
    
    return status


def format_time(seconds):
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_dashboard():
    """Print monitoring dashboard."""
    os.system('clear')
    
    print("="*80)
    print("           POISSON VAE PARAMETER SWEEP - MONITORING DASHBOARD")
    print("="*80)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    status = get_status()
    
    if status is None:
        print("  ⚠ No sweep initialized. Run with --init first.")
        return
    
    total = len(status['pending']) + len(status['running']) + len(status['completed']) + len(status['failed'])
    
    # Summary
    print("  PROGRESS")
    print("  " + "-"*40)
    completed = len(status['completed'])
    pct = (completed / total) * 100 if total > 0 else 0
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  [{bar}] {pct:.1f}%")
    print()
    print(f"  Pending:   {len(status['pending']):3d}")
    print(f"  Running:   {len(status['running']):3d}")
    print(f"  Completed: {len(status['completed']):3d}")
    print(f"  Failed:    {len(status['failed']):3d}")
    print(f"  Total:     {total:3d}")
    print()
    
    # Running jobs
    print("  CURRENTLY RUNNING")
    print("  " + "-"*40)
    if status['running']:
        for job in status['running']:
            gpu = job.get('gpu', '?')
            dist = job['dist_type']
            temp = job['temp']
            anneal = "A" if job['anneal'] else "N"
            n = job['n_param']
            seed = job['seed']
            
            elapsed = ""
            if 'start_time' in job:
                elapsed = format_time(time.time() - job['start_time'])
                elapsed = f" ({elapsed})"
            
            print(f"  GPU {gpu}: {dist}_T{temp}_{anneal}_n{n}_s{seed}{elapsed}")
    else:
        print("  No jobs running")
    print()
    
    # Recent completions
    print("  RECENT COMPLETIONS (last 5)")
    print("  " + "-"*40)
    recent = status['completed'][-5:] if status['completed'] else []
    if recent:
        for job in reversed(recent):
            dist = job['dist_type']
            temp = job['temp']
            anneal = "A" if job['anneal'] else "N"
            n = job['n_param']
            seed = job['seed']
            print(f"  ✓ {dist}_T{temp}_{anneal}_n{n}_s{seed}")
    else:
        print("  None yet")
    print()
    
    # Failed jobs
    if status['failed']:
        print("  FAILED JOBS")
        print("  " + "-"*40)
        for job in status['failed']:
            dist = job['dist_type']
            temp = job['temp']
            anneal = "A" if job['anneal'] else "N"
            n = job['n_param']
            seed = job['seed']
            print(f"  ✗ {dist}_T{temp}_{anneal}_n{n}_s{seed}")
        print()
    
    # GPU utilization (basic)
    print("  GPU STATUS")
    print("  " + "-"*40)
    gpu_jobs = {0: None, 1: None, 2: None, 3: None}
    for job in status['running']:
        gpu = job.get('gpu')
        if gpu is not None:
            gpu_jobs[gpu] = job
    
    for gpu_id in range(4):
        job = gpu_jobs.get(gpu_id)
        if job:
            dist = job['dist_type'][:3]
            print(f"  GPU {gpu_id}: 🟢 Training ({dist})")
        else:
            if len(status['pending']) > 0:
                print(f"  GPU {gpu_id}: 🟡 Idle (waiting for work)")
            else:
                print(f"  GPU {gpu_id}: ⚪ Idle (no more jobs)")
    print()
    
    # ETA
    if status['completed'] and status['running']:
        # Estimate based on completed jobs
        total_completed = len(status['completed'])
        
        # Check for timing info
        times = []
        for job in status['completed']:
            if 'start_time' in job and 'end_time' in job:
                times.append(job['end_time'] - job['start_time'])
        
        if times:
            avg_time = sum(times) / len(times)
            remaining = len(status['pending']) + len(status['running'])
            # With 4 GPUs
            eta_seconds = (remaining / 4) * avg_time
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            print(f"  Estimated completion: {eta.strftime('%Y-%m-%d %H:%M')}")
            print(f"  (avg job time: {format_time(avg_time)})")
    
    print()
    print("="*80)
    print("  Press Ctrl+C to exit monitoring")
    print("="*80)


def main():
    print("Starting sweep monitor...")
    print("Refreshing every 30 seconds...")
    
    try:
        while True:
            print_dashboard()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == '__main__':
    main()
