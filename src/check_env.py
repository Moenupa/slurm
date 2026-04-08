#!/bin/python3

__doc__ = f"""
Checks the installation and availability of PyTorch CUDA components.
Usage:
    python {__file__}
    # torch@2.8.0+cu128(12.8): True
    # flash_attn@2.8.3: True
    # vllm@0.11.0
"""
__author__ = "Meng Wang"
__email__ = "49304833+Moenupa@users.noreply.github.com"


from importlib.metadata import version as pkg_version

from torch.backends.cuda import is_flash_attention_available
from torch.cuda import is_available as cuda_is_available
from torch.version import cuda as torch_cuda_version


def safe_ver(pkg_name) -> str | None:
    try:
        return f"{pkg_name}@{pkg_version(pkg_name)}"
    except Exception:
        return f"{pkg_name}@None"


CHECKLIST = rf"""
{safe_ver("torch")}({torch_cuda_version}): {cuda_is_available()}
{safe_ver("flash_attn")}: {is_flash_attention_available()}
{safe_ver("vllm")}
"""


if __name__ == "__main__":
    print(CHECKLIST.strip())
