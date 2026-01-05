#!/bin/bash
# Monitor training progress for both retraining jobs

echo "================================================================"
echo "  RETRAINING PROGRESS MONITOR"
echo "================================================================"
echo ""

# Function to get last lines from log file
get_last_lines() {
    local dir=$1
    local n=$2
    
    if [ -d "$dir" ]; then
        # Find the most recent log file
        log_file=$(find "$dir" -name "*.txt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            echo "  Last $n lines from log:"
            tail -n $n "$log_file" | sed 's/^/    /'
        else
            echo "  (No log file found yet)"
        fi
    else
        echo "  (Directory not created yet)"
    fi
}

echo "1. SWEEP_ANNEAL (GPU 0) - poisson_temp0.1_annealTrue_n256_seed0_v2"
echo "----------------------------------------------------------------"
get_last_lines "./checkpoints/poisson_temp0.1_annealTrue_n256_seed0_v2" 5
echo ""

echo "2. LINLIN_V2 (GPU 1) - poisson_lin_lin_v2"
echo "----------------------------------------------------------------"
get_last_lines "./checkpoints/poisson_lin_lin_v2" 5
echo ""

echo "================================================================"
echo "GPU Status:"
echo "----------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s - Util: %s%%, Mem: %sMiB/%sMiB\n", $1, $2, $3, $4, $5}'
echo ""

echo "================================================================"
echo "Tmux Sessions:"
echo "----------------------------------------------------------------"
tmux ls 2>/dev/null | grep "retrain" | sed 's/^/  /'
echo ""
echo "To attach to a session:"
echo "  tmux attach -t retrain_sweep   (GPU 0)"
echo "  tmux attach -t retrain_linlin  (GPU 1)"
echo "================================================================"
