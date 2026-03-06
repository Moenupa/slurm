#!/bin/sh
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -t 30-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -J Qwen3-VL-8B-Instruct/Qwen/m3llm

# train model by job name `model/src/dest`, e.g. `Qwen3-8B/Qwen/m3llm`
# Qwen/Qwen3-8B ---data=m3llm--> Qwen-m3llm/Qwen3-8B

# this trajectory == a linked list.
# give proper dataset names to manage this tree of models.

. .venv/bin/activate

set -e
set -x

# META-configs
MODEL_DIR=saves
CONFIG=examples/train_full/qwen3vl_full_sft.yaml

# parse SLURM_JOB_NAME for training
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
SRC=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
DEST=$(DELIM='/' IDX=2 python3 slurm/src/argv.py "$SLURM_JOB_NAME") # registered dataset name

IN=${IN:-"$MODEL_DIR/$SRC/$MODELID"}
OUT=${OUT:-"$MODEL_DIR/$SRC-$DEST/$MODELID"}
DATA=${DATA:-$DEST}

# overrides the META-configs
ARGS=""
ARGS="$ARGS --model=$IN"
ARGS="$ARGS --output_dir=$OUT"
ARGS="$ARGS --dataset=$DATA"

export MODELSCOPE_CACHE=$HOME/.cache/modelscope

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
export NPROC_PER_NODE=$GPUS_PER_NODE

srun --mpi=pmix slurm/multinode/swift.pernode.sh \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	--nnodes=$NNODES \
	--nproc_per_node=$GPUS_PER_NODE \
	--load_from_cache_file true \
	--train_type full \
	--freeze_vit false \
	--freeze_aligner false \
	--freeze_llm false \
	--gradient_checkpointing true \
	--torch_dtype bfloat16 \
	--per_device_train_batch_size 16 \
	--gradient_accumulation_steps 2 \
	--sequence_parallel_size 2 \
	--save_total_limit 3 \
	--eval_strategy "no" \
	--save_only_model true \
	--save_strategy epoch \
	--report_to wandb \
	--num_train_epochs 3 \
	--max_epochs 3 \
	--max_length 32768 \
	--max_new_tokens 1024 \
	--learning_rate 1e-5 \
	--lr_scheduler_type cosine \
	--dataloader_num_workers 28 \
	--dataset_num_proc 28 \
	--deepspeed zero3 \
	--warmup_ratio 0.1 \
	--attn_impl flash_attn \
	--logging_steps 10 \
	--use_logits_to_keep false \
	--padding_free true \
	$ARGS \
	$@
