#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=background
#SBATCH --output=logs/lora.out
#SBATCH --error=logs/lora.err

python lora.py