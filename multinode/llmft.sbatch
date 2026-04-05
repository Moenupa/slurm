#!/bin/sh
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J Qwen3-VL-4B-Instruct/Qwen/m3llm

# train model by job name `model/src/dest`, e.g. `Qwen3-8B/Qwen/m3llm`
# Qwen/Qwen3-8B ---data=m3llm--> Qwen-m3llm/Qwen3-8B

# this trajectory == a linked list.
# give proper dataset names to manage this tree of models.

. .venv/bin/activate

set -e
set -x

export WANDB_NAME=$SLURM_JOB_NAME
export WANDB_RUN_GROUP=llmft

# META-configs
MODEL_DIR=saves
CONFIG=examples/train_full/qwen3vl_full_sft.yaml

# parse SLURM_JOB_NAME for training
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
SRC=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
DEST=$(DELIM='/' IDX=2 python3 slurm/src/argv.py "$SLURM_JOB_NAME") # registered dataset name

# if data is too long, set DATA env var before sbatch
DATA=${DATA:-$DEST}

# overrides the META-configs
ARGS=""
ARGS="$ARGS model_name_or_path=$MODEL_DIR/$SRC/$MODELID"
ARGS="$ARGS output_dir=$MODEL_DIR/$SRC:$DEST/$MODELID"
ARGS="$ARGS dataset=$DATA"
ARGS="$ARGS per_device_train_batch_size=2"
ARGS="$ARGS gradient_accumulation_steps=2"

if command -v scontrol &> /dev/null
then
	export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
else
	export MASTER_ADDR=localhost
fi
export MASTER_PORT=29500
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}
export NNODES=${SLURM_NNODES:-1}
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

srun --mpi=pmix slurm/multinode/llmft.pernode.sh \
	$CONFIG \
	$ARGS \
	$@
