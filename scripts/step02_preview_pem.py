#!/usr/bin/env python3
"""Generate PEM previews for visual QA (a few samples per class)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.env import ensure_conda_env

ensure_conda_env()

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from mpfadet.geometry import CLASS_TO_ID, PNG_H, PNG_W, box_xyxy
from mpfadet.io_iq import list_sample_ids, parse_dr, read_iq_wav
from mpfadet.pem import build_complex, compute_fields, fields_to_pem


def load_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def overlay(rgb: np.ndarray, boxes: list[dict], title: str) -> Image.Image:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font = load_font()
    draw.text((4, 4), title, fill=(255, 255, 0), font=font)
    for b in boxes:
        x0, y0, x1, y1 = box_xyxy(b)
        draw.rectangle([x0, y0, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)], outline=(255, 64, 64), width=2)
        draw.text((x0 + 2, max(0, y0 - 10)), b["Content"], fill=(255, 255, 0), font=font)
    return img


def channel_img(pem: np.ndarray, ch: int) -> np.ndarray:
    g = pem[:, :, ch]
    return np.stack([g, g, g], axis=-1)


def pick_by_class(raw_dir: Path, per_class: int) -> dict[str, list[str]]:
    chosen: dict[str, list[str]] = defaultdict(list)
    for sid in list_sample_ids(raw_dir):
        dr = raw_dir / f"{sid}.DR.txt"
        if not dr.exists():
            continue
        classes = {b["Content"] for b in parse_dr(dr)}
        for c in classes:
            if len(chosen[c]) < per_class:
                chosen[c].append(sid)
        if all(len(v) >= per_class for v in chosen.values()) and set(chosen) >= set(CLASS_TO_ID):
            break
    return dict(chosen)


def box_channel_stats(pem: np.ndarray, boxes: list[dict]) -> dict:
    h, w = pem.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = box_xyxy(b, w, h)
        mask[y0:y1, x0:x1] = True
    out = {}
    names = ["if", "coh", "resid"]
    for i, name in enumerate(names):
        ch = pem[:, :, i].astype(np.float32)
        out[name] = {
            "inside": float(ch[mask].mean()) if mask.any() else 0.0,
            "outside": float(ch[~mask].mean()) if (~mask).any() else 0.0,
        }
        out[name]["contrast"] = out[name]["inside"] - out[name]["outside"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/step02_preview")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/step02_preview_report.json")
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    picked = pick_by_class(args.raw_dir, args.per_class)
    rows = []
    for cls, sids in picked.items():
        for sid in sids:
            iq, fs = read_iq_wav(args.raw_dir / f"{sid}.wav")
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
            mag = np.array(Image.open(args.raw_dir / f"{sid}_spectrogram.png").convert("RGB"))
            boxes = parse_dr(args.raw_dir / f"{sid}.DR.txt")
            stats = box_channel_stats(pem, boxes)
            r, g, b = channel_img(pem, 0), channel_img(pem, 1), channel_img(pem, 2)
            mag_o = overlay(mag, boxes, f"{sid} MAG")
            pem_o = overlay(pem, boxes, f"{sid} PEM cls={cls}")
            strip = Image.new("RGB", (PNG_W * 5, PNG_H))
            for i, im in enumerate([mag_o, pem_o, Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)]):
                strip.paste(im, (i * PNG_W, 0))
            out_path = args.out_dir / f"{cls}_{sid}_strip.png"
            strip.save(out_path)
            rec = {"id": sid, "focus_class": cls, "classes": [b["Content"] for b in boxes], "stats": stats, "file": str(out_path)}
            rows.append(rec)
            print(sid, cls, {k: round(v["contrast"], 2) for k, v in stats.items()})

    args.report.write_text(json.dumps({"iq": {"swap": IQ_SWAP, "conj": IQ_CONJ, "fftshift": FFTSHIFT}, "samples": rows}, indent=2))
    print("wrote", args.report, "n=", len(rows))


if __name__ == "__main__":
    main()
