#!/bin/sh
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p a100
#SBATCH -x compute-a100-01
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=12
#SBATCH -t 30-00:00:00
#SBATCH -J /slurm/models/Qwen2.5-VL-7B-Instruct
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# saved for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

set -x

TF_VER=${TF_VER:-5}
VLLM_VER=${VLLM_VER:-0.16.0}

# run vllm 
uvx --python 3.12 --with "transformers<${TF_VER}" vllm@$VLLM_VER serve \
	$SLURM_JOB_NAME \
	--trust-remote-code \
	--port $SLURM_JOB_ID \
	--served-model-name $(basename $SLURM_JOB_NAME) \
	-dp $SLURM_GPUS_ON_NODE \
	$@
