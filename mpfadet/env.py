"""Force all scripts onto the project conda env."""

from __future__ import annotations

import os
import sys
from pathlib import Path

CONDA_PYTHON = Path("/home/finnwe/miniconda3/envs/mpfadet/bin/python")
CONDA_ENV = "mpfadet"


def ensure_conda_env() -> None:
    want = CONDA_PYTHON.resolve()
    have = Path(sys.executable).resolve()
    if have == want:
        return
    if not want.exists():
        raise RuntimeError(
            f"conda env '{CONDA_ENV}' python not found: {want}. "
            "Create it with: conda create -n mpfadet python=3.12"
        )
    os.execv(str(want), [str(want), *sys.argv])
