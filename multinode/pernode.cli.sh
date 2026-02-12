#!/bin/sh
# NOTE: do not modify this, unless you know what you are doing

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export TOKENIZERS_PARALLELISM=true
export FORCE_TORCHRUN=1

llamafactory-cli train $@