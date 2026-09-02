#!/usr/bin/env python3
"""Train baseline A (mag-only) or full dual MPFADet."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.env import ensure_conda_env

ensure_conda_env()

import torch
from torch.utils.data import DataLoader

from mpfadet.config import ROOT
from mpfadet.dataset import MagPhaseDataset, collate
from mpfadet.eval import decode_batch, map50
from mpfadet.loss import DetectionLoss
from mpfadet.model import MPFADet


def shift_head_for_p2(state: dict) -> dict:
    """Map 3-level fuse/head indices 0,1,2 onto 1,2,3 so P2 slot 0 stays random."""
    out = {}
    prefixes = ("fuse.", "head.loc.", "head.cls.")
    for k, v in state.items():
        nk = k
        for prefix in prefixes:
            if k.startswith(prefix):
                rest = k[len(prefix) :]
                idx, dot, tail = rest.partition(".")
                if idx.isdigit() and dot:
                    nk = f"{prefix}{int(idx) + 1}.{tail}"
                break
        out[nk] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/train_A")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--pem-only", action="store_true", help="baseline B: MagNet on PEM only (no magnitude PNG)")
    parser.add_argument("--p2", action="store_true", help="add stride-2 P2 head for thin boxes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()
    if args.dual and args.pem_only:
        raise SystemExit("--dual and --pem-only are mutually exclusive (B is single-stream MagNet on PEM)")
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ds = MagPhaseDataset(
        args.data, "train", args.imgsz, dual=args.dual, mag_from_phase=args.pem_only
    )
    val_ds = MagPhaseDataset(
        args.data, "val", args.imgsz, dual=args.dual, mag_from_phase=args.pem_only
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate
    )

    model = MPFADet(nc=9, dual=args.dual, p2=args.p2).to(device)
    if args.weights is not None:
        ckpt0 = torch.load(args.weights, map_location=device, weights_only=False)
        sd = ckpt0["model"]
        src_strides = list(ckpt0.get("strides") or [])
        if args.p2 and len(src_strides) == 3:
            sd = shift_head_for_p2(sd)
            print("remapped 3-level fuse/head -> P3/P4/P5 slots")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.weights} missing={len(missing)} unexpected={len(unexpected)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = DetectionLoss(nc=9, strides=model.strides, imgsz=args.imgsz)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"python={sys.executable} device={device} dual={args.dual} p2={args.p2} pem_only={args.pem_only} "
        f"params={n_params/1e6:.2f}M strides={model.strides} train={len(train_ds)} val={len(val_ds)}"
    )

    log_path = args.out / "train.log"
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = {"box": 0.0, "obj": 0.0, "cls": 0.0, "n": 0, "n_pos": 0}
        for mag, phase, targets, _ids in train_loader:
            mag = mag.to(device)
            phase_t = phase.to(device) if phase is not None else None
            preds, _ = model(mag, phase_t)
            loss, parts = criterion(preds, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            run["box"] += parts["box"]
            run["obj"] += parts["obj"]
            run["cls"] += parts["cls"]
            run["n_pos"] += parts["n_pos"]
            run["n"] += 1
        sched.step()
        model.eval()
        vrun = {"box": 0.0, "obj": 0.0, "cls": 0.0, "n": 0, "loss": 0.0}
        all_dets, all_tgts = [], []
        with torch.no_grad():
            for mag, phase, targets, _ids in val_loader:
                mag = mag.to(device)
                phase_t = phase.to(device) if phase is not None else None
                preds, _ = model(mag, phase_t)
                loss, parts = criterion(preds, targets)
                vrun["box"] += parts["box"]
                vrun["obj"] += parts["obj"]
                vrun["cls"] += parts["cls"]
                vrun["loss"] += float(loss.detach())
                vrun["n"] += 1
                if epoch % args.eval_every == 0:
                    dets = decode_batch(preds, model.strides, args.imgsz, conf_th=0.2, iou_th=0.5)
                    all_dets.extend(dets)
                    all_tgts.extend(targets)
        metrics = {"mAP50": None, "mAP75": None}
        if all_dets:
            metrics = map50(all_dets, all_tgts, args.imgsz, nc=9, iou_th=0.5)
            m75 = map50(all_dets, all_tgts, args.imgsz, nc=9, iou_th=0.75)
            metrics["mAP75"] = m75["mAP50"]
        rec = {
            "epoch": epoch,
            "train_box": run["box"] / max(run["n"], 1),
            "train_obj": run["obj"] / max(run["n"], 1),
            "train_cls": run["cls"] / max(run["n"], 1),
            "train_n_pos": run["n_pos"] / max(run["n"], 1),
            "val_box": vrun["box"] / max(vrun["n"], 1),
            "val_obj": vrun["obj"] / max(vrun["n"], 1),
            "val_cls": vrun["cls"] / max(vrun["n"], 1),
            "val_loss": vrun["loss"] / max(vrun["n"], 1),
            "mAP50": metrics.get("mAP50"),
            "mAP75": metrics.get("mAP75"),
            "sec": round(time.time() - t0, 1),
            "lr": sched.get_last_lr()[0],
        }
        history.append(rec)
        map_s = ""
        if rec["mAP50"] is not None:
            map_s = f" mAP50={rec['mAP50']:.3f} mAP75={rec['mAP75']:.3f}"
        line = (
            f"epoch {epoch:03d} train box={rec['train_box']:.3f} obj={rec['train_obj']:.3f} "
            f"cls={rec['train_cls']:.3f} npos={rec['train_n_pos']:.1f} "
            f"val box={rec['val_box']:.3f} obj={rec['val_obj']:.3f} cls={rec['val_cls']:.3f} "
            f"vloss={rec['val_loss']:.3f}{map_s} {rec['sec']}s"
        )
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "rec": rec,
            "strides": list(model.strides),
        }
        torch.save(ckpt, args.out / "last.pt")
        score = 0.0
        if rec["mAP50"] is not None:
            score = rec["mAP50"] + (rec["mAP75"] or 0.0)
        else:
            score = -rec["val_loss"]
        if score > best:
            best = score
            torch.save(ckpt, args.out / "best.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print("best", best, "wrote", args.out)


if __name__ == "__main__":
    main()
