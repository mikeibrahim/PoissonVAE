#!/usr/bin/env python3
"""
Sweep Runner: Manages GPU allocation and launches training jobs from a queue.
Monitors GPU availability and dynamically schedules jobs to maximize utilization.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import itertools

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class GPUMonitor:
    """Monitor GPU memory and utilization."""
    
    def __init__(self, memory_threshold_mb: int = 2000):
        """
        Args:
            memory_threshold_mb: Consider GPU available if free memory > this value (MB)
        """
        self.memory_threshold_mb = memory_threshold_mb
    
    def get_gpu_info(self) -> List[Dict]:
        """Get current GPU status using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                idx, mem_used, mem_total, util = line.split(', ')
                gpus.append({
                    'index': int(idx),
                    'memory_used_mb': int(mem_used),
                    'memory_total_mb': int(mem_total),
                    'memory_free_mb': int(mem_total) - int(mem_used),
                    'utilization_pct': int(util)
                })
            return gpus
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return []
    
    def get_available_gpus(self) -> List[int]:
        """Return list of GPU indices that are available (low memory usage)."""
        gpus = self.get_gpu_info()
        available = []
        for gpu in gpus:
            if gpu['memory_free_mb'] > self.memory_threshold_mb:
                available.append(gpu['index'])
        return available
    
    def print_status(self):
        """Print current GPU status."""
        gpus = self.get_gpu_info()
        print("\n" + "="*70)
        print("GPU STATUS:")
        print("="*70)
        for gpu in gpus:
            status = "✓ AVAILABLE" if gpu['memory_free_mb'] > self.memory_threshold_mb else "✗ BUSY"
            print(f"GPU {gpu['index']}: {gpu['memory_free_mb']:5d} MB free / "
                  f"{gpu['memory_total_mb']:5d} MB total  |  "
                  f"{gpu['utilization_pct']:3d}% util  |  {status}")
        print("="*70 + "\n")


class JobQueue:
    """Manage queue of training jobs."""
    
    def __init__(self, queue_file: str):
        self.queue_file = Path(queue_file)
        self.running_jobs: Dict[int, Dict] = {}  # gpu_id -> job_info
        
    def load_queue(self) -> List[Dict]:
        """Load pending jobs from queue file."""
        if not self.queue_file.exists():
            return []
        
        with open(self.queue_file, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]
    
    def save_queue(self, jobs: List[Dict]):
        """Save remaining jobs to queue file."""
        with open(self.queue_file, 'w') as f:
            for job in jobs:
                f.write(json.dumps(job) + '\n')
    
    def remove_job(self, job: Dict):
        """Remove a job from the queue file."""
        jobs = self.load_queue()
        jobs = [j for j in jobs if j != job]
        self.save_queue(jobs)
    
    def get_num_pending(self) -> int:
        """Get number of pending jobs."""
        return len(self.load_queue())
    
    def get_num_running(self) -> int:
        """Get number of currently running jobs."""
        return len(self.running_jobs)


class SweepRunner:
    """Main sweep runner that schedules and monitors jobs."""
    
    def __init__(
        self,
        queue_file: str,
        checkpoint_dir: str,
        max_jobs_per_gpu: int = 1,
        memory_threshold_mb: int = 2000,
        check_interval_sec: int = 60,
    ):
        self.queue = JobQueue(queue_file)
        self.gpu_monitor = GPUMonitor(memory_threshold_mb)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.check_interval_sec = check_interval_sec
        
        # Track running jobs: {tmux_session_name: (gpu_id, job_dict)}
        self.running_jobs: Dict[str, Tuple[int, Dict]] = {}
    
    def launch_job(self, job: Dict, gpu_id: int) -> str:
        """
        Launch a training job on specified GPU.
        Returns tmux session name.
        """
        # Create unique session name
        session_name = f"sweep_{job['model_type']}_{job['run_id']}_gpu{gpu_id}"
        
        # Build command
        cmd_parts = [
            './scripts/fit_vae.sh',
            str(gpu_id),
            job['dataset'],
            job['model_type'],
            f"'{job['architecture']}'",
            '--verbose'
        ]
        
        # Add all parameters
        for key, value in job['params'].items():
            if value is True:
                cmd_parts.append(f'--{key}')
            elif value is False:
                continue  # Don't add False boolean flags
            else:
                cmd_parts.append(f'--{key}')
                cmd_parts.append(str(value))
        
        # Add comment with sweep info
        comment = f"sweep_{job['model_type']}_seed{job['params']['seed']}_run{job['run_id']}"
        cmd_parts.extend(['--comment', comment])
        
        # Join command
        cmd = ' '.join(cmd_parts)
        
        # Create tmux session
        tmux_cmd = f'tmux new-session -d -s {session_name} "cd {Path.cwd()} && {cmd}"'
        
        print(f"\n🚀 Launching job on GPU {gpu_id}:")
        print(f"   Session: {session_name}")
        print(f"   Model: {job['model_type']}")
        print(f"   Params: {job['params']}")
        print(f"   Command: {cmd}")
        
        try:
            subprocess.run(tmux_cmd, shell=True, check=True)
            self.running_jobs[session_name] = (gpu_id, job)
            return session_name
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to launch job: {e}")
            return None
    
    def check_job_status(self, session_name: str) -> bool:
        """Check if a tmux session is still running. Returns True if running."""
        result = subprocess.run(
            f'tmux has-session -t {session_name} 2>/dev/null',
            shell=True,
            capture_output=True
        )
        return result.returncode == 0
    
    def update_running_jobs(self):
        """Check status of running jobs and update the list."""
        completed = []
        for session_name in list(self.running_jobs.keys()):
            if not self.check_job_status(session_name):
                gpu_id, job = self.running_jobs[session_name]
                print(f"\n✅ Job completed: {session_name} (GPU {gpu_id})")
                print(f"   Model: {job['model_type']}, Seed: {job['params']['seed']}")
                completed.append(session_name)
        
        # Remove completed jobs
        for session_name in completed:
            del self.running_jobs[session_name]
    
    def schedule_jobs(self):
        """Schedule as many jobs as possible given GPU availability."""
        available_gpus = self.gpu_monitor.get_available_gpus()
        pending_jobs = self.queue.load_queue()
        
        if not pending_jobs:
            return
        
        # Count jobs per GPU
        gpu_job_counts = {gpu_id: 0 for gpu_id in available_gpus}
        for session_name, (gpu_id, _) in self.running_jobs.items():
            if gpu_id in gpu_job_counts:
                gpu_job_counts[gpu_id] += 1
        
        # Schedule new jobs
        scheduled = []
        for job in pending_jobs:
            # Find GPU with capacity
            for gpu_id in available_gpus:
                if gpu_job_counts[gpu_id] < self.max_jobs_per_gpu:
                    session_name = self.launch_job(job, gpu_id)
                    if session_name:
                        gpu_job_counts[gpu_id] += 1
                        scheduled.append(job)
                        time.sleep(2)  # Small delay between launches
                    break
        
        # Remove scheduled jobs from queue
        if scheduled:
            remaining = [j for j in pending_jobs if j not in scheduled]
            self.queue.save_queue(remaining)
    
    def run(self):
        """Main loop: monitor GPUs and schedule jobs."""
        print("="*70)
        print("SWEEP RUNNER STARTED")
        print("="*70)
        print(f"Queue file: {self.queue.queue_file}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Max jobs per GPU: {self.max_jobs_per_gpu}")
        print(f"Check interval: {self.check_interval_sec}s")
        print("="*70)
        
        iteration = 0
        while True:
            iteration += 1
            
            # Update status of running jobs
            self.update_running_jobs()
            
            # Print status
            pending = self.queue.get_num_pending()
            running = len(self.running_jobs)
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            print(f"📋 Pending jobs: {pending}")
            print(f"🏃 Running jobs: {running}")
            
            if running > 0:
                print("\nCurrently running:")
                for session_name, (gpu_id, job) in self.running_jobs.items():
                    print(f"  • GPU {gpu_id}: {job['model_type']} (seed={job['params']['seed']})")
            
            # Check GPU status
            self.gpu_monitor.print_status()
            
            # Schedule new jobs if queue not empty
            if pending > 0:
                print("🔄 Scheduling new jobs...")
                self.schedule_jobs()
            elif running == 0:
                print("\n" + "="*70)
                print("✅ ALL JOBS COMPLETED!")
                print("="*70)
                break
            else:
                print("⏳ Waiting for running jobs to complete...")
            
            # Wait before next check
            time.sleep(self.check_interval_sec)
        
        print("\n🎉 Sweep runner finished!")


def generate_sweep_configs(
    dataset: str = 'vH16',
    architecture: str = 'lin|lin',
    temp_stop_values: List[float] = [0.01, 0.1, 0.3],
    temp_anneal_values: List[bool] = [True, False],
    indicator_approx_values: List[str] = ['sigmoid', 'cubic'],
    upperbound_method_values: List[str] = ['fixed'],
    upperbound_param_values: List[int] = [25, 50, 100],
    seed_values: List[int] = [0, 1, 2],
) -> List[Dict]:
    """
    Generate all sweep configurations.
    
    Returns list of job dictionaries with all parameter combinations.
    """
    jobs = []
    run_id = 0
    
    # Generate Poisson (exp) configurations
    for temp_stop, temp_anneal, indicator_approx, seed in itertools.product(
        temp_stop_values,
        temp_anneal_values,
        indicator_approx_values,
        seed_values
    ):
        job = {
            'run_id': run_id,
            'model_type': 'poisson',
            'dataset': dataset,
            'architecture': architecture,
            'params': {
                'temp_stop': temp_stop,
                'temp_anneal': temp_anneal,
                'indicator_approx': indicator_approx,
                'seed': seed,
            }
        }
        jobs.append(job)
        run_id += 1
    
    # Generate GumbelSoftmaxPoisson (GS) configurations
    for temp_stop, temp_anneal, upperbound_method, upperbound_param, seed in itertools.product(
        temp_stop_values,
        temp_anneal_values,
        upperbound_method_values,
        upperbound_param_values,
        seed_values
    ):
        job = {
            'run_id': run_id,
            'model_type': 'gumbel_poisson',
            'dataset': dataset,
            'architecture': architecture,
            'params': {
                'temp_stop': temp_stop,
                'temp_anneal': temp_anneal,
                'upperbound_method': upperbound_method,
                'upperbound_param': upperbound_param,
                'seed': seed,
            }
        }
        jobs.append(job)
        run_id += 1
    
    return jobs


def main():
    parser = argparse.ArgumentParser(description='Sweep Runner for VAE Training')
    parser.add_argument('--generate-queue', action='store_true',
                       help='Generate queue file with all sweep configurations')
    parser.add_argument('--run', action='store_true',
                       help='Run the sweep (monitor and schedule jobs)')
    parser.add_argument('--queue-file', default='sweep_queue.jsonl',
                       help='Path to queue file')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--max-jobs-per-gpu', type=int, default=1,
                       help='Maximum concurrent jobs per GPU')
    parser.add_argument('--memory-threshold', type=int, default=2000,
                       help='GPU free memory threshold (MB) to consider available')
    parser.add_argument('--check-interval', type=int, default=60,
                       help='Seconds between GPU checks')
    parser.add_argument('--dataset', default='vH16',
                       help='Dataset name')
    parser.add_argument('--architecture', default='lin|lin',
                       help='Architecture specification')
    
    args = parser.parse_args()
    
    if args.generate_queue:
        print("Generating sweep configurations...")
        jobs = generate_sweep_configs(
            dataset=args.dataset,
            architecture=args.architecture,
        )
        
        # Save to queue file
        queue_path = Path(args.queue_file)
        with open(queue_path, 'w') as f:
            for job in jobs:
                f.write(json.dumps(job) + '\n')
        
        print(f"\n✅ Generated {len(jobs)} jobs")
        print(f"   Queue file: {queue_path.absolute()}")
        
        # Print summary
        poisson_count = sum(1 for j in jobs if j['model_type'] == 'poisson')
        gumbel_count = sum(1 for j in jobs if j['model_type'] == 'gumbel_poisson')
        print(f"\nBreakdown:")
        print(f"  • Poisson: {poisson_count} configurations")
        print(f"  • GumbelSoftmaxPoisson: {gumbel_count} configurations")
        
    elif args.run:
        runner = SweepRunner(
            queue_file=args.queue_file,
            checkpoint_dir=args.checkpoint_dir,
            max_jobs_per_gpu=args.max_jobs_per_gpu,
            memory_threshold_mb=args.memory_threshold,
            check_interval_sec=args.check_interval,
        )
        runner.run()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Generate queue")
        print("  python scripts/sweep_runner.py --generate-queue")
        print("\n  # Run sweep")
        print("  python scripts/sweep_runner.py --run")


if __name__ == '__main__':
    main()
