#!/bin/bash
#SBATCH --job-name=mlp
#SBATCH --output=logs/mlp.out
#SBATCH --error=logs/mlp.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=30:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

python -m scripts.train_mlp.train_mlp