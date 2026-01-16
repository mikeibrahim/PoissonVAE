#!/bin/bash

# Run 18 GumbelSoftmax adaptive upperbound sweep experiments
# Usage: ./run_adaptive_sweep.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# Get available GPUs (excluding GPU 3 which is busy)
GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}

# Parameters
TEMPS=(0.01 0.1 0.3)
TEMP_ANNEALS=(true false)
SEEDS=(0 1 2)

# Create runs list
declare -a RUNS=()
for temp in "${TEMPS[@]}"; do
    for anneal in "${TEMP_ANNEALS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUNS+=("$temp $anneal $seed")
        done
    done
done

echo "Total runs: ${#RUNS[@]}"
echo "Available GPUs: ${GPUS[*]}"
echo ""

# Function to run a single experiment
run_experiment() {
    local gpu=$1
    local temp=$2
    local anneal=$3
    local seed=$4
    
    # Build command
    local cmd="${ROOT_DIR}/.venv/bin/python -m main.train_vae"
    cmd+=" $gpu"
    cmd+=" vH16"
    cmd+=" gumbel_poisson"
    cmd+=" lin+lin"
    cmd+=" --seed $seed"
    cmd+=" --temp_stop $temp"
    cmd+=" --upperbound_method adaptive"
    cmd+=" --upperbound_param 50"
    cmd+=" --no_wandb"
    
    if [ "$anneal" = "true" ]; then
        cmd+=" --temp_anneal True"
    else
        cmd+=" --temp_anneal False"
    fi
    
    echo "Running on GPU $gpu: temp=$temp, anneal=$anneal, seed=$seed"
    eval "$cmd"
}

# Run experiments sequentially on available GPUs
run_idx=0
for run in "${RUNS[@]}"; do
    read -r temp anneal seed <<< "$run"
    gpu=${GPUS[$((run_idx % NUM_GPUS))]}
    run_experiment "$gpu" "$temp" "$anneal" "$seed"
    ((run_idx++))
done

echo ""
echo "All experiments completed!"
