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
export MODELSCOPE_CACHE=/tmp/ms
export TRITON_HOME=/tmp/triton
export USE_HF=1
export CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE:-2048}
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}

set -e
set -x

# parse SLURM_JOB_NAME for training
MODELID=$(DELIM='/' IDX=0 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
SRC=$(DELIM='/' IDX=1 python3 slurm/src/argv.py "$SLURM_JOB_NAME")
DEST=$(DELIM='/' IDX=2 python3 slurm/src/argv.py "$SLURM_JOB_NAME") # registered dataset name

# second-order overrides
MODEL_DIR=${MODEL_DIR:-saves}
IN=${IN:-"$MODEL_DIR/$SRC/$MODELID"}
OUT=${OUT:-"$MODEL_DIR/$SRC:$DEST/$MODELID"}
DATA=${DATA:-$DEST}

ARGS=""
ARGS="$ARGS --model=$IN"
ARGS="$ARGS --output_dir=$OUT"
ARGS="$ARGS --dataset=$DATA"

EPOCHS=${EPOCHS:-2}

GLOBAL_BS=${GLOBAL_BS:-512}
MICRO_BS=${MICRO_BS:-16}
SP=${SP:-2}
GRAD_ACC=$(expr $GLOBAL_BS \* $SP / $SLURM_NNODES / $SLURM_GPUS_PER_NODE / $MICRO_BS)

srun --mpi=pmix slurm/multinode/swift.pernode.sh \
	--attn_impl flash_attn \
	--dataloader_num_workers 8 \
	--dataset_num_proc 8 \
	--deepspeed zero3 \
	--eval_strategy "no" \
	--freeze_aligner false \
	--freeze_llm false \
	--freeze_vit false \
	--gradient_accumulation_steps $GRAD_ACC \
	--gradient_checkpointing true \
	--learning_rate 1e-5 \
	--vit_lr 2e-6 \
	--load_from_cache_file true \
	--logging_steps 10 \
	--max_epochs $EPOCHS \
	--max_length 32768 \
	--max_new_tokens 1024 \
	--num_train_epochs $EPOCHS \
	--padding_free true \
	--per_device_train_batch_size $MICRO_BS \
	--report_to wandb \
	--save_only_model true \
	--save_strategy epoch \
	--save_total_limit $EPOCHS \
	--sequence_parallel_size $SP \
	--torch_dtype bfloat16 \
	--tuner_type full \
	--use_liger_kernel true \
	--use_logits_to_keep false \
	--vit_gradient_checkpointing true \
	--warmup_ratio 0.1 \
	$ARGS \
	$@
