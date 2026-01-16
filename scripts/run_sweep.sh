#!/bin/bash
# Sweep Management Script
# Provides convenient commands for managing the parameter sweep

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SWEEP_SCRIPT="$SCRIPT_DIR/sweep_runner.py"
QUEUE_FILE="$PROJECT_ROOT/sweep_queue.jsonl"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

cd "$PROJECT_ROOT" || exit 1

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 {generate|run|status|clean|kill}"
    echo ""
    echo "Commands:"
    echo "  generate    Generate sweep queue with all parameter combinations"
    echo "  run         Start sweep runner in tmux (monitors GPUs and launches jobs)"
    echo "  status      Show current sweep status (pending/running jobs)"
    echo "  clean       Clean up queue file"
    echo "  kill        Kill all sweep-related tmux sessions"
    echo ""
    echo "Examples:"
    echo "  $0 generate              # Generate queue"
    echo "  $0 run                   # Start sweep"
    echo "  $0 status                # Check status"
    exit 1
}

generate_queue() {
    echo -e "${BLUE}Generating sweep queue...${NC}"
    python3 "$SWEEP_SCRIPT" --generate-queue \
        --queue-file "$QUEUE_FILE" \
        --checkpoint-dir "$CHECKPOINT_DIR"
    
    if [ -f "$QUEUE_FILE" ]; then
        num_jobs=$(wc -l < "$QUEUE_FILE")
        echo -e "${GREEN}✓ Generated ${num_jobs} jobs${NC}"
        echo -e "${BLUE}Queue file: ${QUEUE_FILE}${NC}"
    fi
}

run_sweep() {
    # Check if queue exists
    if [ ! -f "$QUEUE_FILE" ]; then
        echo -e "${RED}Error: Queue file not found!${NC}"
        echo -e "${YELLOW}Run '$0 generate' first to create the queue.${NC}"
        exit 1
    fi
    
    num_jobs=$(wc -l < "$QUEUE_FILE")
    if [ "$num_jobs" -eq 0 ]; then
        echo -e "${YELLOW}Warning: Queue is empty!${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Starting sweep runner...${NC}"
    echo -e "${BLUE}Queue: ${num_jobs} pending jobs${NC}"
    
    # Check if tmux session already exists
    if tmux has-session -t sweep_runner 2>/dev/null; then
        echo -e "${YELLOW}Sweep runner session already exists!${NC}"
        echo -e "${BLUE}Attach with: tmux attach -t sweep_runner${NC}"
        exit 0
    fi
    
    # Create tmux session
    tmux new-session -d -s sweep_runner "cd '$PROJECT_ROOT' && python3 '$SWEEP_SCRIPT' --run \
        --queue-file '$QUEUE_FILE' \
        --checkpoint-dir '$CHECKPOINT_DIR' \
        --max-jobs-per-gpu 1 \
        --memory-threshold 2000 \
        --check-interval 60"
    
    echo -e "${GREEN}✓ Sweep runner started in tmux session 'sweep_runner'${NC}"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo "  • Attach:  tmux attach -t sweep_runner"
    echo "  • Detach:  Ctrl+B then D"
    echo "  • Status:  $0 status"
}

show_status() {
    echo -e "${BLUE}Sweep Status${NC}"
    echo -e "${BLUE}============${NC}"
    
    # Check queue
    if [ -f "$QUEUE_FILE" ]; then
        num_pending=$(wc -l < "$QUEUE_FILE")
        echo -e "📋 Pending jobs: ${GREEN}${num_pending}${NC}"
    else
        echo -e "📋 Pending jobs: ${YELLOW}0 (no queue file)${NC}"
    fi
    
    # Check running tmux sessions
    echo ""
    echo -e "${BLUE}Running Sessions:${NC}"
    if tmux ls 2>/dev/null | grep -E "sweep_|vae_" > /dev/null; then
        tmux ls 2>/dev/null | grep -E "sweep_|vae_" | while read -r line; do
            echo "  • $line"
        done
    else
        echo "  ${YELLOW}No active sessions${NC}"
    fi
    
    # GPU status
    echo ""
    echo -e "${BLUE}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | \
        awk -F', ' '{
            free = $3 - $2;
            status = (free > 2000) ? "✓ AVAILABLE" : "✗ BUSY";
            printf "  GPU %d: %5d MB free / %5d MB total | %3d%% util | %s\n",
                $1, free, $3, $4, status
        }'
}

clean_queue() {
    if [ -f "$QUEUE_FILE" ]; then
        rm "$QUEUE_FILE"
        echo -e "${GREEN}✓ Queue file deleted${NC}"
    else
        echo -e "${YELLOW}No queue file to clean${NC}"
    fi
}

kill_sessions() {
    echo -e "${RED}Killing all sweep-related tmux sessions...${NC}"
    
    # Kill sweep runner
    if tmux has-session -t sweep_runner 2>/dev/null; then
        tmux kill-session -t sweep_runner
        echo -e "${GREEN}✓ Killed sweep_runner${NC}"
    fi
    
    # Kill individual job sessions
    killed=0
    for session in $(tmux ls 2>/dev/null | grep "sweep_" | cut -d: -f1); do
        tmux kill-session -t "$session" 2>/dev/null && ((killed++)) || true
    done
    
    if [ "$killed" -gt 0 ]; then
        echo -e "${GREEN}✓ Killed ${killed} job sessions${NC}"
    else
        echo -e "${YELLOW}No job sessions to kill${NC}"
    fi
}

# Main command dispatch
case "${1:-}" in
    generate)
        generate_queue
        ;;
    run)
        run_sweep
        ;;
    status)
        show_status
        ;;
    clean)
        clean_queue
        ;;
    kill)
        kill_sessions
        ;;
    *)
        usage
        ;;
esac
