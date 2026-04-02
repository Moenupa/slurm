#!/bin/python3

__doc__ = f"""
Get arg1.split(DELIM)[IDX].
Usage:
    DELIM="," IDX=0 python3 {__file__} "a,b,c" # => "a"

Env Var:
    - DELIM: the delimiter to split, e.g. ','.
    - IDX: index to select after split.
"""
__author__ = "Meng Wang"
__email__ = "49304833+Moenupa@users.noreply.github.com"

import os
import sys


DELIM = os.environ["DELIM"]
IDX = os.environ["IDX"]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        exit(1)

    print(sys.argv[1].split(DELIM)[int(IDX)])
