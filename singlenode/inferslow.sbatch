#!/bin/sh
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=256G
#SBATCH --gres=gpu:1
#SBATCH -J Qwen3-VL-2B-Instruct/trainset/testset

# # ------------------------------------------------------------------------------
# export SLURM_JOB_NAME="modelname/trainset/testset"
# # ------------------------------------------------------------------------------
# # optionally: specify these to override
# export MODEL_IN_DIR=saves/hf
# export TRAIN_CONFIG=examples/train_full/qwen3vl_full_sft.yaml
# export RESULT_SAVE_TO=saves/modelname/trainset/testset


. .venv/bin/activate

set -e
set -x

# META-configs
TRAIN_CONFIG=${TRAIN_CONFIG:-'examples/train_full/qwen3vl_full_sft.yaml'}

# parse SLURM_JOB_NAME for training
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TRAINSET=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TESTSET=$(DELIM='/' IDX=2 python3 slurm/src/argv.py "$SLURM_JOB_NAME")

# input paths
MODEL_IN_DIR=${MODEL_IN_DIR:-"saves/$TRAINSET"}
BASE_ARGS="model_name_or_path=$MODEL_IN_DIR/$MODELID dataset=$TESTSET eval_dataset=$TESTSET"

# output paths
RESULT_SAVE_TO=${RESULT_SAVE_TO:-"saves/$MODELID/$TRAINSET/$TESTSET"}

llamafactory-cli train $TRAIN_CONFIG \
	$BASE_ARGS \
	do_train=false do_predict=true predict_with_generate=true report_to=none deepspeed='' \
	$@ \
	output_dir=$RESULT_SAVE_TO
