#!/bin/bash
# Launch all remaining training jobs

cd /home/michael/code/wandb_dataset

echo "=========================================="
echo "Launching Training Jobs"
echo "=========================================="

# 1. Resume incomplete Poisson run 30
echo ""
echo "1. Resuming incomplete Poisson run (seed0_run30)..."
tmux new-session -d -s resume_run30 "cd /home/michael/code/wandb_dataset && .venv/bin/python -m main.resume_train 0 'poisson_uniform_c(-4)_vH16_z-512_fp_nrm-none_<lin|lin>' 'sweep_poisson_seed0_run30_mc_b1000-ep3000-lr(0.005)_beta(1:0x0.5)_gr(500)_(2026_01_09,00:23)' --verbose"
echo "   ✓ Started: resume_run30"

# 2. Launch GumbelSoftmax sweep with wandb disabled
echo ""
echo "2. Launching GumbelSoftmax sweep (54 jobs)..."
tmux new-session -d -s gs_sweep "cd /home/michael/code/wandb_dataset && python3 scripts/sweep_runner.py --run --queue-file sweep_gs_queue.jsonl"
echo "   ✓ Started: gs_sweep"

echo ""
echo "=========================================="
echo "All jobs launched!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  tmux attach -t resume_run30    # Watch Poisson resume"
echo "  tmux attach -t gs_sweep        # Watch GS sweep runner"
echo "  tmux ls                         # List all sessions"
echo ""
