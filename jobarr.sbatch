#!/bin/sh
#SBATCH -o logs/%x/%A_%a.out
#SBATCH -e logs/%x/%A_%a.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --gres=gpu:1
#SBATCH -J jobarr
#SBATCH --array=0-0

# for env debugging
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# python src/check_env.py

set -e
set -x

# TODO: run here
