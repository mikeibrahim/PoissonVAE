#!/bin/bash
# Launch both retraining jobs in tmux sessions

# Kill existing sessions if they exist
tmux kill-session -t retrain_sweep 2>/dev/null
tmux kill-session -t retrain_linlin 2>/dev/null

echo "Starting training jobs in tmux..."

# Start sweep_anneal training on GPU 0
tmux new-session -d -s retrain_sweep \
    "cd /home/michael/code/poisson_notebook && \
     source .venv/bin/activate && \
     python retrain_models.py --model sweep_anneal --gpu 0; \
     echo 'Training complete! Press any key to close...'; \
     read"

echo "✓ Started sweep_anneal training on GPU 0 (session: retrain_sweep)"

# Start linlin_v2 training on GPU 1
tmux new-session -d -s retrain_linlin \
    "cd /home/michael/code/poisson_notebook && \
     source .venv/bin/activate && \
     python retrain_models.py --model linlin_v2 --gpu 1; \
     echo 'Training complete! Press any key to close...'; \
     read"

echo "✓ Started linlin_v2 training on GPU 1 (session: retrain_linlin)"

echo ""
echo "Training sessions started!"
echo "To monitor sweep_anneal (GPU 0):  tmux attach -t retrain_sweep"
echo "To monitor linlin_v2 (GPU 1):     tmux attach -t retrain_linlin"
echo ""
echo "To detach from a session: Press Ctrl+B then D"
echo "To list all sessions:     tmux ls"
