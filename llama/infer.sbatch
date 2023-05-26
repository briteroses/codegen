#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=background
#SBATCH --output=logs/cases.out
#SBATCH --error=logs/cases.err
#SBATCH --array=2

python inference.py --exparray $SLURM_ARRAY_TASK_ID