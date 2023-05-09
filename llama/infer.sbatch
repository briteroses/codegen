#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --partition=urop
#SBATCH --output=logs/demo.out
#SBATCH --error=logs/demo.err

python inference_alpaca.py