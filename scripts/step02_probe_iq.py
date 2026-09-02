#!/usr/bin/env python3
"""Probe I/Q channel order / conjugate / fftshift by matching STFT |X| to PNG boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.env import ensure_conda_env

ensure_conda_env()

import numpy as np
from PIL import Image

from mpfadet.geometry import PNG_H, PNG_W, box_xyxy
from mpfadet.io_iq import list_sample_ids, parse_dr, read_iq_wav
from mpfadet.pem import build_complex, compute_fields, logmag_image


def corr2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-12:
        return 0.0
    return float((a @ b) / d)


def box_energy_score(logmag_img: np.ndarray, boxes: list[dict]) -> float:
    gray = logmag_img[:, :, 0].astype(np.float32)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = box_xyxy(b, w, h)
        mask[y0:y1, x0:x1] = True
    if not mask.any() or mask.all():
        return 0.0
    return float(gray[mask].mean() - gray[~mask].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/home/finnwe/project/paper/few_shot_data"))
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/finnwe/project/paper/MPFADet/logs/step02_iq_probe.json"),
    )
    args = parser.parse_args()

    ids = list_sample_ids(args.raw_dir)
    pick = [
        "background_2_2493",
        "background_5_8",
        "background_2_54",
        "background_5_461",
        "background_2_219",
        "background_5_81",
    ]
    pick = [s for s in pick if s in ids][: args.n]
    while len(pick) < min(args.n, len(ids)):
        for s in ids:
            if s not in pick:
                pick.append(s)
            if len(pick) >= args.n:
                break

    configs = []
    for swap in (False, True):
        for conj in (False, True):
            for fftshift in (False, True):
                configs.append({"swap": swap, "conj": conj, "fftshift": fftshift})

    rows = []
    for sid in pick:
        wav = args.raw_dir / f"{sid}.wav"
        png = args.raw_dir / f"{sid}_spectrogram.png"
        dr = args.raw_dir / f"{sid}.DR.txt"
        iq, fs = read_iq_wav(wav)
        rgb = np.array(Image.open(png).convert("RGB"))
        gray = rgb[:, :, 0].astype(np.float32)
        boxes = parse_dr(dr)
        rec = {"id": sid, "fs": fs, "n": int(iq.shape[0]), "configs": []}
        print(f"== {sid} fs={fs} n={iq.shape[0]} ch_std={iq.std(0).tolist()} ch_corr={float(np.corrcoef(iq[:,0], iq[:,1])[0,1]):.3f}")
        for cfg in configs:
            x = build_complex(iq, cfg["swap"], cfg["conj"])
            fields = compute_fields(x, args.n_fft, args.hop, float(fs), cfg["fftshift"])
            img = logmag_image(fields["logmag"], PNG_H, PNG_W, flip_freq=True)
            c = corr2(img[:, :, 0], gray)
            s = box_energy_score(img, boxes)
            rec["configs"].append({**cfg, "corr_png": c, "box_contrast": s})
            print(
                f"  swap={int(cfg['swap'])} conj={int(cfg['conj'])} shift={int(cfg['fftshift'])} "
                f"corr={c:.3f} box_contrast={s:.2f}"
            )
        rows.append(rec)

    # rank configs by mean corr + contrast
    summary = []
    for cfg in configs:
        corrs = []
        cons = []
        for rec in rows:
            for r in rec["configs"]:
                if r["swap"] == cfg["swap"] and r["conj"] == cfg["conj"] and r["fftshift"] == cfg["fftshift"]:
                    corrs.append(r["corr_png"])
                    cons.append(r["box_contrast"])
        summary.append(
            {
                **cfg,
                "mean_corr": float(np.mean(corrs)),
                "mean_box_contrast": float(np.mean(cons)),
                "score": float(np.mean(corrs) + 0.01 * np.mean(cons)),
            }
        )
    summary.sort(key=lambda d: d["score"], reverse=True)
    best = summary[0]
    report = {"n_fft": args.n_fft, "hop": args.hop, "ids": pick, "best": best, "summary": summary, "per_id": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("BEST", best)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
