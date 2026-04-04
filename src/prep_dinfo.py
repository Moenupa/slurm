#!/bin/python3

__doc__ = f"""
prepare data/dataset_info.json for dataset registration.

Usage:
    python {__file__} path/to/your.json
    # create a new json at arg and symlink it to data/dataset_info.json

Env Var:
    - TARGET: the target json file to write, default to data/dataset_info.json
    - FORMAT: choices: alpaca, sharegpt, openai, default to sharegpt.
"""
__author__ = "Meng Wang"
__email__ = "49304833+Moenupa@users.noreply.github.com"

import json
import os
import subprocess
import sys


TARGET = os.getenv("TARGET", "data/dataset_info.json")
FORMAT: str = os.getenv("FORMAT", "sharegpt")
assert FORMAT in ["alpaca", "sharegpt", "openai"], f"Unsupported FORMAT: {FORMAT}"

HF = os.environ["HF"].split(",")
SUBSET = os.getenv("SUBSET", "default").split(",")
SPLIT = os.getenv("SPLIT", "train").split(",")


def get_columns(formatting: str) -> dict:
    match formatting:
        case "sharegpt":
            column_mapping = {
                "messages": "conversations",
                "images": "images",
            }
        case "openai":
            column_mapping = {
                "messages": "messages",
                "images": "images",
            }
        case "alpaca":
            column_mapping = {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "images": "images",
            }
        case _:
            return {}
    return {"columns": column_mapping}


def get_tags(formatting: str) -> dict:
    match formatting:
        case "sharegpt":
            return {}
            tag_mapping = {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system",
            }
        case "openai":
            tag_mapping = {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system",
            }
        case _:
            return {}
    return {"tags": tag_mapping}


def get_formatting(formatting: str) -> dict:
    return {"formatting": {"openai": "sharegpt"}.get(formatting, formatting)}


def get_dset_info(
    hf_path: str,
    subset: str = "default",
    split: str = "train",
) -> dict:
    return (
        {
            "hf_hub_url": hf_path,
            "subset": subset,
            "split": split,
        }
        | get_formatting(FORMAT)
        | get_columns(FORMAT)
        | get_tags(FORMAT)
    )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(__doc__)
        exit(1)

    d_to_write = {}
    for hf_path in HF:
        dname = os.path.basename(hf_path)

        for subset in SUBSET:
            for split in SPLIT:
                entry = f"{dname}_{subset}_{split}".lower()
                d_to_write[entry] = get_dset_info(hf_path, subset, split)

    if len(sys.argv) == 1:
        print(json.dumps(d_to_write, indent=4, ensure_ascii=False))
        exit(0)

    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(d_to_write, f, indent=4, ensure_ascii=False)

    subprocess.run(["rm", "data/dataset_info.json"], check=False)
    subprocess.run(
        ["ln", "-s", os.path.basename(sys.argv[1]), "data/dataset_info.json"],
        check=False,
    )
