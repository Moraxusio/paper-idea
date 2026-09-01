#!/usr/bin/env python3
"""Train baseline A (mag-only) or full dual MPFADet."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpfadet.config import ROOT
from mpfadet.dataset import MagPhaseDataset, collate
from mpfadet.loss import DetectionLoss
from mpfadet.model import MPFADet


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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ds = MagPhaseDataset(args.data, "train", args.imgsz, dual=args.dual)
    val_ds = MagPhaseDataset(args.data, "val", args.imgsz, dual=args.dual)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate
    )

    model = MPFADet(nc=9, dual=args.dual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = DetectionLoss(nc=9, imgsz=args.imgsz)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} dual={args.dual} params={n_params/1e6:.2f}M train={len(train_ds)} val={len(val_ds)}")

    log_path = args.out / "train.log"
    history = []
    best = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = {"box": 0.0, "obj": 0.0, "cls": 0.0, "n": 0}
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
            run["n"] += 1
        sched.step()
        model.eval()
        vrun = {"box": 0.0, "obj": 0.0, "cls": 0.0, "n": 0, "loss": 0.0}
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
        rec = {
            "epoch": epoch,
            "train_box": run["box"] / max(run["n"], 1),
            "train_obj": run["obj"] / max(run["n"], 1),
            "train_cls": run["cls"] / max(run["n"], 1),
            "val_box": vrun["box"] / max(vrun["n"], 1),
            "val_obj": vrun["obj"] / max(vrun["n"], 1),
            "val_cls": vrun["cls"] / max(vrun["n"], 1),
            "val_loss": vrun["loss"] / max(vrun["n"], 1),
            "sec": round(time.time() - t0, 1),
            "lr": sched.get_last_lr()[0],
        }
        history.append(rec)
        line = (
            f"epoch {epoch:03d} train box={rec['train_box']:.3f} obj={rec['train_obj']:.3f} cls={rec['train_cls']:.3f} "
            f"val box={rec['val_box']:.3f} obj={rec['val_obj']:.3f} cls={rec['val_cls']:.3f} "
            f"vloss={rec['val_loss']:.3f} {rec['sec']}s"
        )
        print(line)
        with log_path.open("a") as f:
            f.write(line + "\n")
        ckpt = {"epoch": epoch, "model": model.state_dict(), "args": vars(args), "rec": rec}
        torch.save(ckpt, args.out / "last.pt")
        if rec["val_loss"] < best:
            best = rec["val_loss"]
            torch.save(ckpt, args.out / "best.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print("best val_loss", best, "wrote", args.out)


if __name__ == "__main__":
    main()
