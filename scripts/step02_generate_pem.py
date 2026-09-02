#!/usr/bin/env python3
"""Batch-generate PEM phase images for every sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.env import ensure_conda_env

ensure_conda_env()

import numpy as np
from PIL import Image

from mpfadet.config import (
    FFTSHIFT,
    FLIP_FREQ,
    GATE_ALPHA,
    GATE_PERCENTILE,
    HOP,
    IQ_CONJ,
    IQ_SWAP,
    N_FFT,
    RAW_DIR,
    ROOT,
)
from mpfadet.geometry import PNG_H, PNG_W
from mpfadet.io_iq import list_sample_ids, read_iq_wav
from mpfadet.pem import build_complex, compute_fields, fields_to_pem


def generate_one(raw_dir: Path, sid: str, out_path: Path) -> dict:
    t0 = time.time()
    iq, fs = read_iq_wav(raw_dir / f"{sid}.wav")
    x = build_complex(iq, IQ_SWAP, IQ_CONJ)
    fields = compute_fields(x, N_FFT, HOP, float(fs), FFTSHIFT)
    pem = fields_to_pem(
        fields,
        PNG_H,
        PNG_W,
        flip_freq=FLIP_FREQ,
        gate_alpha=GATE_ALPHA,
        gate_percentile=GATE_PERCENTILE,
        hop=HOP,
        n_fft=N_FFT,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pem).save(out_path)
    return {
        "id": sid,
        "shape": list(pem.shape),
        "mean": [float(pem[:, :, i].mean()) for i in range(3)],
        "sec": round(time.time() - t0, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/phase")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/step02_generate_report.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = list_sample_ids(args.raw_dir)
    if args.limit:
        ids = ids[: args.limit]
    t_all = time.time()
    rows = []
    failed = []
    for i, sid in enumerate(ids, 1):
        out_path = args.out_dir / f"{sid}.png"
        if args.skip_existing and out_path.exists():
            continue
        try:
            rec = generate_one(args.raw_dir, sid, out_path)
            rows.append(rec)
        except Exception as e:
            failed.append({"id": sid, "error": str(e)})
            print("FAIL", sid, e)
            continue
        if i % 50 == 0 or i == 1 or i == len(ids):
            elapsed = time.time() - t_all
            rate = i / max(elapsed, 1e-6)
            eta = (len(ids) - i) / max(rate, 1e-6)
            print(f"[{i}/{len(ids)}] {sid} {rec['sec']:.3f}s mean={rec['mean']} eta={eta/60:.1f}min")

    report = {
        "n_ids": len(ids),
        "n_ok": len(rows),
        "n_fail": len(failed),
        "failed": failed,
        "elapsed_sec": round(time.time() - t_all, 2),
        "out_dir": str(args.out_dir),
        "settings": {
            "n_fft": N_FFT,
            "hop": HOP,
            "swap": IQ_SWAP,
            "conj": IQ_CONJ,
            "fftshift": FFTSHIFT,
            "flip_freq": FLIP_FREQ,
        },
        "mean_rgb": (
            np.mean([r["mean"] for r in rows], axis=0).tolist() if rows else None
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print("DONE", report["n_ok"], "fail", report["n_fail"], "sec", report["elapsed_sec"])
    print("wrote", args.report)


if __name__ == "__main__":
    main()
