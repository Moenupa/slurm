#!/bin/sh
# NOTE: do not modify this, unless you know what you are doing

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NODE_RANK=${SLURM_NODEID:-0}
export TOKENIZERS_PARALLELISM=true
export FORCE_TORCHRUN=1

llamafactory-cli train $@
