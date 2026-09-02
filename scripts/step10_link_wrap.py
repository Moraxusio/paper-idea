#!/usr/bin/env python3
"""Symlink wrap PNGs into the same train/val/test ids as magnitude images."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.env import ensure_conda_env

ensure_conda_env()

from mpfadet.config import ROOT


def ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    processed = ROOT / "data/processed"
    wrap_src = processed / "wrap"
    missing = 0
    n = 0
    for split in ("train", "val", "test"):
        mag_dir = processed / "images" / split
        out_dir = processed / "wrap" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for mag in sorted(mag_dir.glob("*.png")):
            src = wrap_src / f"{mag.stem}.png"
            if not src.exists():
                missing += 1
                continue
            ensure_link(src, out_dir / mag.name)
            n += 1
    print(f"linked {n} wrap images, missing {missing}")


if __name__ == "__main__":
    main()
