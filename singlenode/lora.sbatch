#!/bin/sh
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=128G
#SBATCH --gres=gpu:1
#SBATCH -J Qwen3-VL-2B-Instruct/trainset/testset

# # ------------------------------------------------------------------------------
# export SLURM_JOB_NAME="modelname/trainset/testset"
# export CLI_TRAIN=1 # to enable training
# export CLI_TEST=1  # to enable testing
# # ------------------------------------------------------------------------------
# # optionally: specify these to override
# export MODEL_IN_DIR=saves/hf
# export TRAIN_CONFIG=examples/train_lora/qwen3vl_lora_sft.yaml
# export LORA_SAVE_TO=saves/modelname/trainset/trainset
# export RESULT_SAVE_TO=saves/modelname/trainset/testset
# # and run this script with shell
# sh slurm/singlenode/lora.sbatch # --epochs 3


. .venv/bin/activate
export DISABLE_VERSION_CHECK=1

set -e
set -x

# META-configs
MODEL_IN_DIR=${MODEL_IN_DIR:-'saves/hf'}
TRAIN_CONFIG=${TRAIN_CONFIG:-'examples/train_lora/qwen3vl_lora_sft.yaml'}

# parse SLURM_JOB_NAME for training
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TRAINSET=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TESTSET=$(DELIM='/' IDX=2 python3 slurm/src/argv.py "$SLURM_JOB_NAME")

# input paths
BASE_ARGS="model_name_or_path=$MODEL_IN_DIR/$MODELID dataset=$TRAINSET eval_dataset=$TESTSET"

# output paths
MODEL_OUT_DIR=saves/$MODELID
TRAINSET_PATH=$(echo $TRAINSET | tr ',' '-')
TESTSET_PATH=$(echo $TESTSET | tr ',' '-')
LORA_SAVE_TO=${LORA_SAVE_TO:-"$MODEL_OUT_DIR/$TRAINSET_PATH/$TRAINSET_PATH"}
RESULT_SAVE_TO=${RESULT_SAVE_TO:-"$MODEL_OUT_DIR/$TRAINSET_PATH/$TESTSET_PATH"}

# you need CLI_TRAIN=1 or CLI_TEST=1 to enable training or testing
if [ "${CLI_TRAIN:-0}" -ne 0 ]; then
	llamafactory-cli train $TRAIN_CONFIG \
		$BASE_ARGS \
		do_train=true do_predict=false report_to=wandb \
		mix_strategy=interleave_once \
		save_strategy=epoch plot_loss=false overwrite_output_dir=false save_only_model=true \
		output_dir=$LORA_SAVE_TO \
		$@
fi
if [ "${CLI_TEST:-0}" -ne 0 ]; then
	llamafactory-cli train $TRAIN_CONFIG \
		$BASE_ARGS \
		do_train=false do_predict=true predict_with_generate=true report_to=none \
		adapter_name_or_path=$LORA_SAVE_TO \
		output_dir=$RESULT_SAVE_TO \
		$@
fi
