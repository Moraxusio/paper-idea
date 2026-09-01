#!/usr/bin/env python3
"""Step 1: map DR.txt boxes onto spectrogram PNGs and decide axis convention."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

T_MAX = 5.0e7
F_MAX = 5.0e4
PNG_W = 875
PNG_H = 656

CLASS_COLORS = {
    "2FSK": (255, 64, 64),
    "4FSK": (255, 140, 0),
    "8-Tone": (255, 215, 0),
    "16-Tone": (50, 205, 50),
    "GMSK": (0, 206, 209),
    "FM": (30, 144, 255),
    "AM-DSB": (138, 43, 226),
    "Morse": (255, 20, 147),
    "PSK": (255, 255, 255),
}


def parse_dr(path: Path) -> list[dict]:
    boxes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            boxes.append(json.loads(line))
    return boxes


def box_xyxy(box: dict, w: int, h: int, y_up: bool) -> tuple[int, int, int, int]:
    x0 = box["DateTimeStart"] / T_MAX * w
    x1 = box["DateTimeEnd"] / T_MAX * w
    if y_up:
        y0 = (1.0 - box["FreqU"] / F_MAX) * h
        y1 = (1.0 - box["FreqD"] / F_MAX) * h
    else:
        y0 = box["FreqD"] / F_MAX * h
        y1 = box["FreqU"] / F_MAX * h
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return (
        int(np.clip(np.floor(xa), 0, w - 1)),
        int(np.clip(np.floor(ya), 0, h - 1)),
        int(np.clip(np.ceil(xb), 1, w)),
        int(np.clip(np.ceil(yb), 1, h)),
    )


def region_stats(gray: np.ndarray, boxes_xyxy: list[tuple[int, int, int, int]]) -> dict:
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    areas = []
    insides = []
    for x0, y0, x1, y1 in boxes_xyxy:
        if x1 <= x0 or y1 <= y0:
            continue
        patch = gray[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] = True
        areas.append(int(patch.size))
        insides.append(float(patch.mean()) if patch.size else 0.0)
    inside = gray[mask]
    outside = gray[~mask]
    return {
        "n_boxes": len(boxes_xyxy),
        "inside_mean": float(inside.mean()) if inside.size else 0.0,
        "outside_mean": float(outside.mean()) if outside.size else 0.0,
        "inside_std": float(inside.std()) if inside.size else 0.0,
        "outside_std": float(outside.std()) if outside.size else 0.0,
        "inside_frac": float(mask.mean()),
        "box_area_mean": float(np.mean(areas)) if areas else 0.0,
        "box_inside_means": insides,
        "contrast": float(inside.mean() - outside.mean()) if inside.size and outside.size else 0.0,
    }


def draw_overlay(rgb: np.ndarray, boxes: list[dict], y_up: bool, title: str) -> Image.Image:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    h, w = rgb.shape[:2]
    draw.text((4, 4), title, fill=(255, 255, 0), font=font)
    for box in boxes:
        x0, y0, x1, y1 = box_xyxy(box, w, h, y_up=y_up)
        color = CLASS_COLORS.get(box["Content"], (255, 0, 0))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
        label = box["Content"]
        draw.text((x0 + 2, max(y0 - 10, 0)), label, fill=color, font=font)
    return img


def list_ids(raw_dir: Path) -> list[str]:
    ids = []
    for png in sorted(raw_dir.glob("*_spectrogram.png")):
        ids.append(png.name[: -len("_spectrogram.png")])
    return ids


def pick_samples(ids: list[str], raw_dir: Path, k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    by_bg = {"background_2": [], "background_5": []}
    class_hits = {c: [] for c in CLASS_COLORS}
    for sid in ids:
        prefix = "_".join(sid.split("_")[:2])
        if prefix in by_bg:
            by_bg[prefix].append(sid)
        dr = raw_dir / f"{sid}.DR.txt"
        if not dr.exists():
            continue
        for box in parse_dr(dr):
            class_hits.setdefault(box["Content"], []).append(sid)
    chosen = []
    for prefix, pool in by_bg.items():
        if pool:
            chosen.extend(rng.sample(pool, min(max(k // 4, 1), len(pool))))
    for cls, pool in class_hits.items():
        uniq = list(dict.fromkeys(pool))
        if uniq:
            chosen.append(rng.choice(uniq))
    rest = [s for s in ids if s not in chosen]
    rng.shuffle(rest)
    chosen.extend(rest)
    out, seen = [], set()
    for sid in chosen:
        if sid in seen:
            continue
        if not (raw_dir / f"{sid}.DR.txt").exists():
            continue
        seen.add(sid)
        out.append(sid)
        if len(out) >= k:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/home/finnwe/project/paper/few_shot_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/finnwe/project/paper/MPFADet/outputs/step01_alignment"))
    parser.add_argument("--report", type=Path, default=Path("/home/finnwe/project/paper/MPFADet/logs/step01_alignment_report.json"))
    parser.add_argument("--n-overlay", type=int, default=24)
    parser.add_argument("--n-stats", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    ids = list_ids(args.raw_dir)
    overlay_ids = pick_samples(ids, args.raw_dir, args.n_overlay, args.seed)
    rng = random.Random(args.seed + 1)
    stats_ids = overlay_ids + [s for s in ids if s not in overlay_ids]
    rng.shuffle(stats_ids)
    stats_ids = stats_ids[: args.n_stats]

    print(f"total png ids={len(ids)}")
    print(f"overlay n={len(overlay_ids)} stats n={len(stats_ids)}")

    per_image = []
    contrast_up, contrast_down = [], []
    class_counter = Counter()
    size_ok = True

    for sid in stats_ids:
        png_path = args.raw_dir / f"{sid}_spectrogram.png"
        dr_path = args.raw_dir / f"{sid}.DR.txt"
        img = Image.open(png_path).convert("RGB")
        rgb = np.array(img)
        h, w = rgb.shape[:2]
        if (w, h) != (PNG_W, PNG_H):
            size_ok = False
        gray = rgb[:, :, 0].astype(np.float32)
        boxes = parse_dr(dr_path)
        for b in boxes:
            class_counter[b["Content"]] += 1
        xyxy_up = [box_xyxy(b, w, h, True) for b in boxes]
        xyxy_dn = [box_xyxy(b, w, h, False) for b in boxes]
        st_up = region_stats(gray, xyxy_up)
        st_dn = region_stats(gray, xyxy_dn)
        contrast_up.append(st_up["contrast"])
        contrast_down.append(st_dn["contrast"])
        rec = {
            "id": sid,
            "wh": [w, h],
            "n_boxes": len(boxes),
            "up": {k: st_up[k] for k in ("inside_mean", "outside_mean", "contrast", "inside_frac")},
            "down": {k: st_dn[k] for k in ("inside_mean", "outside_mean", "contrast", "inside_frac")},
        }
        per_image.append(rec)

    mean_up = float(np.mean(contrast_up))
    mean_dn = float(np.mean(contrast_down))
    win_up = int(sum(u > d for u, d in zip(contrast_up, contrast_down)))
    decision = "y_up" if mean_up >= mean_dn else "y_down"
    y_up = decision == "y_up"

    for sid in overlay_ids:
        png_path = args.raw_dir / f"{sid}_spectrogram.png"
        dr_path = args.raw_dir / f"{sid}.DR.txt"
        rgb = np.array(Image.open(png_path).convert("RGB"))
        boxes = parse_dr(dr_path)
        a = draw_overlay(rgb, boxes, True, f"{sid}  y_up (f increases up)")
        b = draw_overlay(rgb, boxes, False, f"{sid}  y_down (f increases down)")
        canvas = Image.new("RGB", (a.width * 2, a.height))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        canvas.save(args.out_dir / f"{sid}_both.png")
        winner = draw_overlay(rgb, boxes, y_up, f"{sid}  chosen={decision}")
        winner.save(args.out_dir / f"{sid}_chosen.png")

    report = {
        "png_expected": [PNG_W, PNG_H],
        "all_png_size_ok": size_ok,
        "n_png": len(ids),
        "n_stats": len(stats_ids),
        "n_overlay": len(overlay_ids),
        "class_counts_in_stats": dict(class_counter),
        "contrast_y_up_mean": mean_up,
        "contrast_y_down_mean": mean_dn,
        "n_images_up_better": win_up,
        "n_images_down_better": len(stats_ids) - win_up,
        "decision": decision,
        "mapping": {
            "time": "x = DateTime / 5e7 * W, DateTime in [0, 5e7] <-> 5s",
            "freq": (
                "y_up: y=(1-Freq/5e4)*H  (0 Hz at bottom)"
                if y_up
                else "y_down: y=Freq/5e4*H  (0 Hz at top)"
            ),
            "crop": "identity, full 875x656 is TF canvas",
        },
        "overlay_ids": overlay_ids,
        "per_image": per_image,
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"contrast y_up={mean_up:.3f} y_down={mean_dn:.3f}")
    print(f"images up better={win_up}/{len(stats_ids)}")
    print(f"DECISION {decision}")
    print(f"wrote {args.report}")
    print(f"overlays {args.out_dir}")


if __name__ == "__main__":
    main()
