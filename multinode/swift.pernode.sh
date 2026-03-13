#!/bin/sh
# NOTE: do not modify this, unless you know what you are doing

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${SLURM_NNODES:-1}
export NODE_RANK=${SLURM_NODEID:-0}
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-8}

export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
export NPROC_PER_NODE=$GPUS_PER_NODE

export TOKENIZERS_PARALLELISM=true

torchrun \
    --master_port $MASTER_PORT \
    --master_addr $MASTER_ADDR \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    -m swift.cli.sft \
    $@ 
