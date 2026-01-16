# Parameter Sweep System

Automated parameter sweep with GPU-aware job scheduling for VAE training.

## Features

- **Automatic GPU Management**: Monitors GPU availability and schedules jobs dynamically
- **Queue-based System**: All configurations pre-generated, jobs run as GPUs become available
- **Checkpoint Saving**: Each model saved to `/checkpoints` with configs and weights
- **WandB Integration**: All runs logged to Weights & Biases
- **Tmux Integration**: Runs in background, persistent across disconnections

## Quick Start

```bash
# 1. Generate sweep queue (all parameter combinations)
./scripts/run_sweep.sh generate

# 2. Start the sweep runner (runs in tmux)
./scripts/run_sweep.sh run

# 3. Check status
./scripts/run_sweep.sh status

# 4. Attach to sweep runner (optional)
tmux attach -t sweep_runner
# Detach: Ctrl+B then D
```

## Sweep Parameters

### Configuration

The sweep covers the following parameters:

**Both Poisson & GumbelSoftmaxPoisson:**
- `temp_stop`: [0.01, 0.1, 0.3] - Final temperature for annealing
- `temp_anneal`: [True, False] - Enable/disable temperature annealing
- `seed`: [0, 1, 2] - Random seeds for reproducibility

**Poisson only:**
- `indicator_approx`: [sigmoid, cubic] - Soft indicator function type

**GumbelSoftmaxPoisson only:**
- `upperbound_method`: [fixed] - Method for categorical upperbound
- `upperbound_param`: [25, 50, 100] - Upperbound parameter values

### Total Configurations

- **Poisson**: 3 (temp_stop) × 2 (temp_anneal) × 2 (indicator) × 3 (seed) = **36 runs**
- **GumbelSoftmax**: 3 (temp_stop) × 2 (temp_anneal) × 3 (upperbound) × 3 (seed) = **54 runs**
- **Total**: **90 training runs**

## GPU Scheduling

The sweep runner automatically:
1. Checks GPU memory availability (threshold: 2000 MB free)
2. Launches 1 job per available GPU
3. Monitors running jobs and reschedules when GPUs free up
4. Continues until all jobs complete

### Resource Limits

- **Max jobs per GPU**: 1 (configurable)
- **Memory threshold**: 2000 MB free required
- **Check interval**: 60 seconds

## Output Structure

### Checkpoints

Each model is saved to `checkpoints/` with the following structure:

```
checkpoints/
├── poisson_<params>_<timestamp>/
│   ├── ConfigPoisVAE.json          # Model config
│   ├── ConfigTrainVAE.json         # Training config
│   ├── PoissonVAE+TrainerVAE-<epoch>.pt  # Checkpoint
│   └── run_info.txt                # Training summary
└── gumbel_poisson_<params>_<timestamp>/
    ├── ConfigGumbelPoisVAE.json
    ├── ConfigTrainVAE.json
    ├── GumbelPoissonVAE+TrainerVAE-<epoch>.pt
    └── run_info.txt
```

### WandB Logs

All runs are logged to WandB with:
- Run name: `sweep_<model>_seed<N>_run<ID>`
- Project: `PoissonVAE` (default)
- Metrics: loss, KL, reconstruction, temperature, etc.

## Management Commands

```bash
# Generate queue
./scripts/run_sweep.sh generate

# Start sweep (in tmux)
./scripts/run_sweep.sh run

# Check status
./scripts/run_sweep.sh status

# Clean queue file
./scripts/run_sweep.sh clean

# Kill all sweep sessions
./scripts/run_sweep.sh kill
```

## Monitoring

### Check sweep status
```bash
./scripts/run_sweep.sh status
```

Output shows:
- Number of pending jobs
- Running tmux sessions
- GPU availability

### Attach to sweep runner
```bash
tmux attach -t sweep_runner
```

Inside tmux:
- See real-time scheduling decisions
- GPU status updates
- Job launch/completion messages
- Detach: `Ctrl+B` then `D`

### View individual job
```bash
# List all sweep job sessions
tmux ls | grep sweep_

# Attach to specific job
tmux attach -t sweep_poisson_123_gpu0
```

## Advanced Usage

### Custom Sweep Parameters

Edit `scripts/sweep_runner.py` and modify `generate_sweep_configs()`:

```python
def generate_sweep_configs(...):
    # Modify parameter values
    temp_stop_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    seed_values = [0, 1, 2, 3, 4]
    # ...
```

Then regenerate queue:
```bash
./scripts/run_sweep.sh clean
./scripts/run_sweep.sh generate
```

### Manual Queue Management

```bash
# View queue
cat sweep_queue.jsonl | jq .

# Count pending jobs
wc -l sweep_queue.jsonl

# Edit queue (remove specific jobs)
vim sweep_queue.jsonl
```

### Adjust GPU Scheduling

Edit sweep runner parameters in `scripts/run_sweep.sh`:

```bash
# Change max jobs per GPU
--max-jobs-per-gpu 2

# Adjust memory threshold (MB)
--memory-threshold 3000

# Change check interval (seconds)
--check-interval 30
```

## Troubleshooting

### Jobs not starting
```bash
# Check GPU availability
nvidia-smi

# Verify queue exists
cat sweep_queue.jsonl

# Check sweep runner is running
tmux ls | grep sweep_runner

# View runner logs
tmux attach -t sweep_runner
```

### Job failed
```bash
# Find the job's tmux session
tmux ls | grep sweep_

# Attach to see error
tmux attach -t sweep_<model>_<id>_gpu<N>

# Check WandB for partial results
# Visit: https://wandb.ai/<entity>/PoissonVAE
```

### Restart sweep
```bash
# Kill all sessions
./scripts/run_sweep.sh kill

# Restart
./scripts/run_sweep.sh run
```

### Queue stuck
```bash
# Clean and regenerate
./scripts/run_sweep.sh clean
./scripts/run_sweep.sh generate
./scripts/run_sweep.sh run
```

## Implementation Details

### Flow

1. **Generate**: Creates `sweep_queue.jsonl` with all configurations
2. **Schedule**: Sweep runner monitors GPUs every 60s
3. **Launch**: When GPU available, launches job in tmux
4. **Monitor**: Tracks job completion, updates queue
5. **Repeat**: Continues until queue empty

### Job Lifecycle

```
[Queue] → [GPU Available] → [Launch in Tmux] → [Train] → [Save Checkpoint] → [Complete]
```

### Checkpoint Auto-saving

The training script automatically saves:
- Model config on initialization
- Training config before training starts
- Checkpoints every N epochs (configurable)
- Final checkpoint on completion

## Files

- `scripts/sweep_runner.py` - Main sweep orchestrator
- `scripts/run_sweep.sh` - Convenience wrapper script
- `sweep_queue.jsonl` - Job queue (auto-generated)
- `checkpoints/` - Output directory for models

## Dependencies

- Python 3.8+
- PyTorch
- WandB
- tmux
- nvidia-smi (CUDA toolkit)

All Python dependencies are in `requirements.txt`.
