#!/bin/python3

__doc__ = f"""
prepare data/dataset_info.json for dataset registration.

Usage:
    python {__file__}

Env Var:
    - TARGET: the target json file to write, default to data/dataset_info.json
    - MOFRL: the model name to run, default to Qwen3-VL-4B-Instruct
    - CLI_TRAIN: whether to include train set, default to 1 (true)
    - CLI_TEST: whether to include test set, default to 1 (true)
    - SRC: the sbatch script to run, default to slurm/singlenode/lora.sbatch
"""
__author__ = "Meng Wang"
__email__ = "49304833+Moenupa@users.noreply.github.com"

import os
import sys
import json


MODEL = os.getenv("MODEL", "Qwen3-VL-4B-Instruct")
TARGET = os.getenv("TARGET", "data/dataset_info.json")
CLI_TRAIN = os.getenv("CLI_TRAIN", "1")
CLI_TEST = os.getenv("CLI_TEST", "1")
SRC = os.getenv("SRC", "slurm/singlenode/lora.sbatch")

with open(TARGET, "r", encoding="utf-8") as f:
    dataset_info: dict[str, dict] = json.load(f)
    assert isinstance(dataset_info, dict), (
        f"Expected dict in {TARGET}, got {type(dataset_info)}"
    )

    ENTRY: set[str] = set(dataset_info.keys())  # ty:ignore[invalid-assignment]
    ENTRY_SET = set(
        e.replace("_train", "").replace("_val", "").replace("_test", "") for e in ENTRY
    )


def get_train_test_set(dataset: str) -> tuple[str, str]:
    train = f"{dataset}_train"
    test = f"{dataset}_val"
    if CLI_TEST == "1" or test not in ENTRY:
        test = f"{dataset}_test"
    return train, test


ENVVAR = f"CLI_TRAIN={CLI_TRAIN} CLI_TEST={CLI_TEST}"
JOBNAME = "{0}/{1}/{2}"


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(__doc__)
        exit(1)

    for _d in ENTRY_SET:
        _1, _2 = get_train_test_set(_d)

        print(ENVVAR, "sbatch -J", JOBNAME.format(MODEL, _1, _2), SRC)
