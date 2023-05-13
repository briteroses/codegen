#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=background
#SBATCH --output=logs/evals_%a.out
#SBATCH --error=logs/evals_%a.err
#SBATCH --array=1-6

python inference.py --exparray $SLURM_ARRAY_TASK_ID