#!/bin/sh
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=64G
#SBATCH --gres=gpu:1
#SBATCH -J Qwen3-VL-2B-Instruct/testset


export DISABLE_VERSION_CHECK=1

set -e
set -x

MODEL_IN_DIR=${MODEL_IN_DIR:-'saves/hf'}
RESULT_DIR=${RESULT_DIR:-'saves/hf_results'}

# parse SLURM_JOB_NAME for testing
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TESTSET=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")

RESULT_SAVE_TO=${RESULT_SAVE_TO:-"$RESULT_DIR/$MODELID/$TESTSET"}

mkdir -p $RESULT_SAVE_TO
uv run scripts/vllm_infer.py \
	--model_name_or_path $MODEL_IN_DIR/$MODELID \
	--dataset $TESTSET \
	--template "qwen3_vl_nothink" \
	--cutoff_len 30720 \
	--max_new_tokens 2048 \
	--enable_thinking false \
	--temperature 0.7 \
	--top_k 20 \
	--top_p 0.8 \
	--save_name $RESULT_SAVE_TO/generated_predictions.jsonl \
	$@
