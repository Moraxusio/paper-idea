#!/usr/bin/env python3
"""Evaluate a checkpoint on val/test: mAP50, mAP75, per-class AP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.env import ensure_conda_env

ensure_conda_env()

import torch
from torch.utils.data import DataLoader

from mpfadet.config import ROOT
from mpfadet.dataset import MagPhaseDataset, collate
from mpfadet.eval import decode_batch, map50
from mpfadet.geometry import ID_TO_CLASS
from mpfadet.model import MPFADet


@torch.no_grad()
def run(ckpt_path: Path, split: str, imgsz: int, batch: int, conf_th: float) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    dual = bool(args.get("dual", "train_F" in str(ckpt_path) or "dual" in str(ckpt_path)))
    if "dual" in args:
        dual = bool(args["dual"])
    strides = tuple(ckpt.get("strides") or (4, 8, 16))
    p2 = bool(args.get("p2", False)) or len(strides) == 4
    pem_only = bool(args.get("pem_only", False))
    phase_subdir = str(args.get("phase_subdir") or "image")
    fusion = str(args.get("fusion") or "gated")
    phase_mask = str(args.get("phase_mask") or "111")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPFADet(nc=9, dual=dual, p2=p2, fusion=fusion)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    ds = MagPhaseDataset(
        ROOT / "data/processed",
        split,
        imgsz,
        dual=dual,
        mag_from_phase=pem_only,
        phase_subdir=phase_subdir,
        phase_mask=phase_mask,
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, collate_fn=collate)
    all_dets, all_tgts = [], []
    for mag, phase, targets, _ids in loader:
        mag = mag.to(device)
        phase_t = phase.to(device) if phase is not None else None
        preds, _ = model(mag, phase_t)
        dets = decode_batch(preds, strides, imgsz, conf_th=conf_th, iou_th=0.5)
        all_dets.extend(dets)
        all_tgts.extend(targets)
    m50 = map50(all_dets, all_tgts, imgsz, nc=9, iou_th=0.5)
    m75 = map50(all_dets, all_tgts, imgsz, nc=9, iou_th=0.75)
    names = {int(k): ID_TO_CLASS[int(k)] for k in range(9)}
    per50 = {names[c]: round(v, 4) for c, v in m50["AP"].items()}
    per75 = {names[c]: round(v, 4) for c, v in m75["AP"].items()}
    n_gt = {names[c]: int(m50["n_gt"][c]) for c in range(9)}
    return {
        "ckpt": str(ckpt_path),
        "split": split,
        "dual": dual,
        "p2": p2,
        "pem_only": pem_only,
        "phase_subdir": phase_subdir,
        "fusion": fusion,
        "phase_mask": phase_mask,
        "epoch": ckpt.get("epoch"),
        "n_images": len(ds),
        "mAP50": round(m50["mAP50"], 4),
        "mAP75": round(m75["mAP50"], 4),
        "AP50": per50,
        "AP75": per75,
        "n_gt": n_gt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    rec = run(args.ckpt, args.split, args.imgsz, args.batch, args.conf)
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print("wrote", args.report)


if __name__ == "__main__":
    main()
