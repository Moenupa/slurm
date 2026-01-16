#!/bin/sh                                                                                                                                                                                                                                         
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=60G
#SBATCH --gres=gpu:1
#SBATCH -J Qwen3-VL-2B-Instruct/trainset/testset

. .venv/bin/activate

set -e
set -x

# META-configs
MODEL_IN_DIR=saves/Qwen
MODEL_OUT_DIR=saves/tmp
TRAIN_CONFIG=examples/train_lora/qwen3vl_lora_sft.yaml

# parse SLURM_JOB_NAME for training
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TRAINSET=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
TESTSET=$(DELIM='/' IDX=2 python3 slurm/src/argv.py "$SLURM_JOB_NAME")

# constant paths
BASE_ARGS="model_name_or_path=$MODEL_IN_DIR/$MODELID"
LORA_SAVE_TO=$MODEL_OUT_DIR/$TRAINSET/$TRAINSET
RESULT_SAVE_TO=$MODEL_OUT_DIR/$TRAINSET/$TESTSET

# you need CLI_TRAIN=1 or CLI_TEST=1 to enable training or testing
if [ "${CLI_TRAIN:-0}" -ne 0 ]; then
    llamafactory-cli train $TRAIN_CONFIG \
        $BASE_ARGS \
        dataset="$TRAINSET" \
        output_dir=$LORA_SAVE_TO \
        $@
fi
if [ "${CLI_TEST:-0}" -ne 0 ]; then
    EXTRA_INFER_ARGS='do_train=false do_predict=true predict_with_generate=true report_to=none'
    llamafactory-cli train $TRAIN_CONFIG \
        $BASE_ARGS \
        $EXTRA_INFER_ARGS \
        adapter_name_or_path=$LORA_SAVE_TO \
        eval_dataset="$TESTSET" \
        output_dir=$RESULT_SAVE_TO \
        $@
fi