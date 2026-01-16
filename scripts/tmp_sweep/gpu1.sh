#!/bin/bash
cd /home/michael/code/wandb_dataset

echo "=== Running gs_adaptive_t001_ann_seed1 on GPU 1 ==="
/home/michael/code/wandb_dataset/.venv/bin/python -m main.train_vae 1 vH16 gumbel_poisson "lin|lin" --seed 1 --temp_stop 0.01 --upperbound_method adaptive --upperbound_param 50 --temp_anneal 2>&1 | tee -a checkpoints/sweep/gs_adaptive_t001_ann_seed1/TrainerVAE.log

echo "=== Running gs_adaptive_t001_noann_seed1 on GPU 1 ==="
/home/michael/code/wandb_dataset/.venv/bin/python -m main.train_vae 1 vH16 gumbel_poisson "lin|lin" --seed 1 --temp_stop 0.01 --upperbound_method adaptive --upperbound_param 50 2>&1 | tee -a checkpoints/sweep/gs_adaptive_t001_noann_seed1/TrainerVAE.log

echo "=== Running gs_adaptive_t01_ann_seed1 on GPU 1 ==="
/home/michael/code/wandb_dataset/.venv/bin/python -m main.train_vae 1 vH16 gumbel_poisson "lin|lin" --seed 1 --temp_stop 0.1 --upperbound_method adaptive --upperbound_param 50 --temp_anneal 2>&1 | tee -a checkpoints/sweep/gs_adaptive_t01_ann_seed1/TrainerVAE.log

echo "=== Running gs_adaptive_t01_noann_seed1 on GPU 1 ==="
/home/michael/code/wandb_dataset/.venv/bin/python -m main.train_vae 1 vH16 gumbel_poisson "lin|lin" --seed 1 --temp_stop 0.1 --upperbound_method adaptive --upperbound_param 50 2>&1 | tee -a checkpoints/sweep/gs_adaptive_t01_noann_seed1/TrainerVAE.log

echo "=== Running gs_adaptive_t03_ann_seed1 on GPU 1 ==="
/home/michael/code/wandb_dataset/.venv/bin/python -m main.train_vae 1 vH16 gumbel_poisson "lin|lin" --seed 1 --temp_stop 0.3 --upperbound_method adaptive --upperbound_param 50 --temp_anneal 2>&1 | tee -a checkpoints/sweep/gs_adaptive_t03_ann_seed1/TrainerVAE.log

echo "=== Running gs_adaptive_t03_noann_seed1 on GPU 1 ==="
/home/michael/code/wandb_dataset/.venv/bin/python -m main.train_vae 1 vH16 gumbel_poisson "lin|lin" --seed 1 --temp_stop 0.3 --upperbound_method adaptive --upperbound_param 50 2>&1 | tee -a checkpoints/sweep/gs_adaptive_t03_noann_seed1/TrainerVAE.log
