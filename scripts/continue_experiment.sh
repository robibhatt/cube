#!/bin/bash
#SBATCH --job-name=continue_experiment
#SBATCH --output=logs/continue_experiment.out
#SBATCH --error=logs/continue_experiment.err
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=20:00:00

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

python -m scripts.continue_experiment.continue_experiment