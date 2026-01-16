# Sweep System - Quick Reference

## ✅ What's Been Created

### 1. Sweep Runner (`scripts/sweep_runner.py`)
- Monitors GPU availability (checks free memory > 2000 MB)
- Dynamically schedules jobs from queue
- Runs 1 job per available GPU
- Launches each job in its own tmux session
- Continues until all 90 jobs complete

### 2. Management Script (`scripts/run_sweep.sh`)
Convenient wrapper with commands:
- `generate` - Create queue with all 90 configurations
- `run` - Start sweep runner in tmux
- `status` - Show pending/running jobs and GPU status
- `clean` - Remove queue file
- `kill` - Stop all sweep sessions

### 3. New Training Flag (`--temp_anneal`)
Added to `main/train_vae.py`:
- `--temp_anneal` flag enables temperature annealing
- When not provided (or explicitly False), sets `temp_anneal_portion = 0.0`
- When True, uses default annealing schedule

## 🚀 How to Run the Sweep

```bash
# Step 1: Generate all 90 configurations
./scripts/run_sweep.sh generate

# Step 2: Start the sweep runner (runs in tmux background)
./scripts/run_sweep.sh run

# Step 3: Monitor progress (optional)
./scripts/run_sweep.sh status

# Or attach to the runner to see live updates
tmux attach -t sweep_runner
# Press Ctrl+B then D to detach
```

## 📊 Sweep Configuration

### Parameters Being Swept

**Poisson (36 runs):**
- temp_stop: [0.01, 0.1, 0.3]
- temp_anneal: [True, False]
- indicator_approx: [sigmoid, cubic]
- seed: [0, 1, 2]

**GumbelSoftmaxPoisson (54 runs):**
- temp_stop: [0.01, 0.1, 0.3]
- temp_anneal: [True, False]
- upperbound_method: [fixed]
- upperbound_param: [25, 50, 100]
- seed: [0, 1, 2]

**Total: 90 training runs**

## 📁 Output Structure

Each model saved to `checkpoints/` with:
```
checkpoints/
└── <model>_<timestamp>/
    ├── Config<Model>VAE.json       # Model config
    ├── ConfigTrainVAE.json         # Training config
    └── <Model>VAE+TrainerVAE-<epoch>.pt  # Checkpoint
```

## 🔍 Monitoring

### Check what's running
```bash
./scripts/run_sweep.sh status
```

Shows:
- Pending jobs count
- Running tmux sessions
- GPU availability (✓/✗)

### Watch sweep runner live
```bash
tmux attach -t sweep_runner
```

You'll see:
- GPU status checks every 60s
- Job launches (with full command)
- Job completions
- Scheduling decisions

### View specific training job
```bash
# List all job sessions
tmux ls | grep sweep_

# Attach to specific job
tmux attach -t sweep_poisson_0_gpu0
```

## 🛠️ Management

### Pause sweep
```bash
# Kill runner (jobs keep running)
tmux kill-session -t sweep_runner

# Resume later
./scripts/run_sweep.sh run
```

### Stop everything
```bash
# Kill all sweep sessions (runner + jobs)
./scripts/run_sweep.sh kill
```

### Modify queue
```bash
# View queue
cat sweep_queue.jsonl | head -5

# Edit manually (remove/add jobs)
vim sweep_queue.jsonl

# Or regenerate
./scripts/run_sweep.sh clean
./scripts/run_sweep.sh generate
```

## 💡 How It Works

1. **Queue Generation**: Creates `sweep_queue.jsonl` with all 90 job configs
2. **GPU Monitoring**: Checks `nvidia-smi` every 60s for free memory
3. **Job Scheduling**: Launches jobs on available GPUs (1 per GPU)
4. **Execution**: Each job runs `fit_vae.sh` with specified parameters
5. **Completion**: When job finishes, GPU freed for next job in queue
6. **Checkpointing**: Each job saves model + configs to `checkpoints/`
7. **WandB Logging**: All metrics logged with unique run names

## 🎯 Current Status

After generation, you now have:
- ✅ 90 jobs in queue (`sweep_queue.jsonl`)
- ✅ 2 GPUs available (0, 1)
- ✅ 2 old jobs still running (vae_cubic_01, vae_cubic_02)
- ⏳ Ready to start sweep with `./scripts/run_sweep.sh run`

## 🔧 Customization

### Change parameters
Edit `scripts/sweep_runner.py`, function `generate_sweep_configs()`:
```python
temp_stop_values: List[float] = [0.01, 0.1, 0.3]  # Modify here
```

### Adjust GPU limits
Edit `scripts/run_sweep.sh`, in `run_sweep()` function:
```bash
--max-jobs-per-gpu 1      # Jobs per GPU
--memory-threshold 2000   # MB free required
--check-interval 60       # Seconds between checks
```

## 📝 Notes

- All jobs use WandB (ensure logged in: `wandb login`)
- Each job runs in isolated tmux session
- Checkpoints auto-saved based on training config
- Safe to disconnect - runs in background via tmux
- Queue persists - can restart runner anytime
- Old running jobs (vae_cubic_01/02) won't interfere

## 🚨 Troubleshooting

**Jobs not starting:**
```bash
# Check runner is active
tmux ls | grep sweep_runner

# If not, start it
./scripts/run_sweep.sh run
```

**GPU says busy but looks free:**
```bash
# Adjust threshold
# Edit scripts/run_sweep.sh, change --memory-threshold value
```

**Want to run fewer jobs:**
```bash
# Edit queue file
vim sweep_queue.jsonl
# Delete unwanted lines, save
```

**Job crashed:**
```bash
# It will be removed from queue automatically
# Check WandB for partial results
# Or re-add to queue manually
```

---

## Next Steps

**Start the sweep:**
```bash
./scripts/run_sweep.sh run
```

**Then monitor:**
```bash
# Quick status
./scripts/run_sweep.sh status

# Or watch live
tmux attach -t sweep_runner
```

The sweep will run unattended, utilizing all available GPUs until all 90 jobs complete! 🎉
