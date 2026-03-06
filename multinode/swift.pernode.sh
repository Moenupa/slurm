#!/bin/sh
# NOTE: do not modify this, unless you know what you are doing

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NODE_RANK=${SLURM_NODEID:-0}
export TOKENIZERS_PARALLELISM=true

torchrun --node_rank=$NODE_RANK -m swift.cli.sft $@
