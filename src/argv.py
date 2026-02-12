#!/bin/python3

__doc__ = f"""
Get arg1.split(DELIM)[IDX]. E.g.:
    DELIM="," IDX=0 python3 {__file__} "a,b,c"
    => "a"
Raises:
    KeyError: if DELIM or IDX not set in env.
    IndexError: if IDX out of range.
"""
__author__ = "Meng Wang"
__email__ = "49304833+Moenupa@users.noreply.github.com"

import os
import sys

DELIM = os.environ["DELIM"]
IDX = os.environ["IDX"]

if __name__ == "__main__":
    print(sys.argv[1].split(DELIM)[int(IDX)])
