#!/usr/bin/env python3
"""Convert DR.txt to YOLO labels and build stratified train/val/test dirs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.config import RAW_DIR, ROOT
from mpfadet.geometry import CLASS_TO_ID, box_yolo
from mpfadet.io_iq import list_sample_ids, parse_dr


def bg_of(sid: str) -> str:
    return "_".join(sid.split("_")[:2])


def stratified_split(ids: list[str], seed: int, n_val: int, n_test: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    by_bg: dict[str, list[str]] = {}
    for sid in ids:
        by_bg.setdefault(bg_of(sid), []).append(sid)
    train, val, test = [], [], []
    n = len(ids)
    for bg, pool in sorted(by_bg.items()):
        rng.shuffle(pool)
        n_bg = len(pool)
        n_test_bg = int(round(n_bg * n_test / n))
        n_val_bg = int(round(n_bg * n_val / n))
        test.extend(pool[:n_test_bg])
        val.extend(pool[n_test_bg : n_test_bg + n_val_bg])
        train.extend(pool[n_test_bg + n_val_bg :])
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_label(dr_path: Path, out_path: Path) -> int:
    boxes = parse_dr(dr_path)
    lines = []
    for b in boxes:
        cls, x, y, w, h = box_yolo(b)
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--phase-dir", type=Path, default=ROOT / "data/processed/phase")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--report", type=Path, default=ROOT / "logs/step03_split_report.json")
    args = parser.parse_args()

    ids = list_sample_ids(args.raw_dir)
    missing_phase = [s for s in ids if not (args.phase_dir / f"{s}.png").exists()]
    splits = stratified_split(ids, args.seed, args.n_val, args.n_test)

    class_counts = {sp: Counter() for sp in splits}
    bg_counts = {sp: Counter() for sp in splits}
    n_boxes = {sp: 0 for sp in splits}

    for split, sids in splits.items():
        mag_dir = args.out_root / "images" / split
        phase_dir = args.out_root / "image" / split
        lab_dir = args.out_root / "labels" / split
        mag_dir.mkdir(parents=True, exist_ok=True)
        phase_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for sid in sids:
            mag_src = args.raw_dir / f"{sid}_spectrogram.png"
            phase_src = args.phase_dir / f"{sid}.png"
            ensure_link(mag_src, mag_dir / f"{sid}.png")
            if phase_src.exists():
                ensure_link(phase_src, phase_dir / f"{sid}.png")
            n = write_label(args.raw_dir / f"{sid}.DR.txt", lab_dir / f"{sid}.txt")
            n_boxes[split] += n
            bg_counts[split][bg_of(sid)] += 1
            for b in parse_dr(args.raw_dir / f"{sid}.DR.txt"):
                class_counts[split][b["Content"]] += 1

    names = [k for k, _ in sorted(CLASS_TO_ID.items(), key=lambda kv: kv[1])]
    yaml_text = "\n".join(
        [
            f"path: {args.out_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "train_ir: image/train",
            "val_ir: image/val",
            "test_ir: image/test",
            "",
            f"nc: {len(names)}",
            "names:",
            *[f"    {i}: {n}" for i, n in enumerate(names)],
            "",
        ]
    )
    yaml_path = args.out_root / "data_MP_Multimodel.yaml"
    yaml_path.write_text(yaml_text)

    report = {
        "n_ids": len(ids),
        "missing_phase": missing_phase[:50],
        "n_missing_phase": len(missing_phase),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "bg_counts": {k: dict(v) for k, v in bg_counts.items()},
        "class_counts": {k: dict(v) for k, v in class_counts.items()},
        "n_boxes": n_boxes,
        "yaml": str(yaml_path),
        "seed": args.seed,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("split_sizes", "bg_counts", "n_boxes", "n_missing_phase")}, indent=2))
    print("wrote", args.report)


if __name__ == "__main__":
    main()
