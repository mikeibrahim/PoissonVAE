#!/bin/bash
# Continuously monitor both training sessions

while true; do
    clear
    echo "================================================================"
    echo "  CONTINUOUS RETRAINING MONITOR - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    echo ""
    
    # Check sweep_anneal
    echo "1. SWEEP_ANNEAL (GPU 0) - Last 5 lines:"
    echo "----------------------------------------------------------------"
    tmux capture-pane -t retrain_sweep -p 2>/dev/null | tail -5 | sed 's/^/  /'
    echo ""
    
    # Check linlin_v2
    echo "2. LINLIN_V2 (GPU 1) - Last 5 lines:"
    echo "----------------------------------------------------------------"
    tmux capture-pane -t retrain_linlin -p 2>/dev/null | tail -5 | sed 's/^/  /'
    echo ""
    
    # GPU status
    echo "================================================================"
    echo "GPU Status:"
    echo "----------------------------------------------------------------"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s - Util: %s%%, Mem: %sMiB/%sMiB, Temp: %sC\n", $1, $2, $3, $4, $5, $6}'
    echo ""
    
    # Check if sessions still exist
    echo "================================================================"
    echo "Session Status:"
    echo "----------------------------------------------------------------"
    if tmux has-session -t retrain_sweep 2>/dev/null; then
        echo "  ✓ retrain_sweep is running"
    else
        echo "  ✗ retrain_sweep has stopped!"
    fi
    
    if tmux has-session -t retrain_linlin 2>/dev/null; then
        echo "  ✓ retrain_linlin is running"
    else
        echo "  ✗ retrain_linlin has stopped!"
    fi
    echo ""
    
    echo "================================================================"
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing every 10 seconds..."
    echo "================================================================"
    
    sleep 10
done
