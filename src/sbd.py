#!/bin/python3

__doc__ = f"""
Sbatch with automatic linkedlist dependency handling.
Usage:
    EXE=echo python {__file__} <<EOF
        echo 1
        echo 2
        echo 3
        EOF

Env Var:
    - AFTER: dependency type (default: 'afterok').
    - FIRST: dependency for the first job
    - ALL: dependency for all jobs (overrides FIRST and subsequent dependencies)
    - EXE: executable to use for submission (default: 'sbatch').
"""
__author__ = "Meng Wang"
__email__ = "49304833+Moenupa@users.noreply.github.com"

import os
import subprocess
import sys


AFTER = os.getenv("AFTER", "afterok")
EXE = os.getenv("EXE", "sbatch")

FIRST = os.getenv("FIRST", None)
ALL = os.getenv("ALL", None)


def submit_one(cmd: list[str], dependency: int | str | None = None) -> int:
    cmd = [c for c in cmd if c not in (EXE, "sbatch", "sb")]

    if dependency is not None:
        cmd = ["-d", f"{AFTER}:{dependency}"] + cmd

    cmd = [EXE] + cmd
    result = subprocess.run(cmd, capture_output=True)
    stdout = result.stdout.decode().strip()
    stderr = result.stderr.decode().strip()

    if result.returncode != 0:
        raise RuntimeError(
            f"Error {result.returncode} submitting {' '.join(cmd)}\nError:{stderr}"
        )

    print("+", " ".join(cmd), file=sys.stderr)
    print(stdout)
    return int(stdout.split()[-1])


if __name__ == "__main__":
    # if no stdin, print usage and exit
    if sys.stdin.isatty():
        print(__doc__)
        exit(1)

    dependency = FIRST
    for line in sys.stdin:
        dependency = submit_one(line.split(), ALL or dependency)
