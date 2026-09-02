from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def letterbox_rgb(path: Path, size: int) -> tuple[torch.Tensor, float, int, int, int, int]:
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    scale = min(size / w0, size / h0)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(img, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor, scale, pad_x, pad_y, w0, h0


def shift_yolo(labels: torch.Tensor, pad_x: int, pad_y: int, nw: int, nh: int, size: int) -> torch.Tensor:
    if labels.numel() == 0:
        return labels
    out = labels.clone()
    out[:, 1] = labels[:, 1] * nw / size + pad_x / size
    out[:, 2] = labels[:, 2] * nh / size + pad_y / size
    out[:, 3] = labels[:, 3] * nw / size
    out[:, 4] = labels[:, 4] * nh / size
    return out


class MagPhaseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        imgsz: int = 512,
        dual: bool = False,
        mag_from_phase: bool = False,
        phase_subdir: str = "image",
    ):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.dual = dual
        self.mag_from_phase = mag_from_phase
        self.phase_subdir = phase_subdir
        mag_dir = self.root / ("image" if mag_from_phase else "images") / split
        self.ids = sorted(p.stem for p in mag_dir.glob("*.png"))
        if not self.ids:
            raise FileNotFoundError(f"no images in {mag_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        mag_rel = "image" if self.mag_from_phase else "images"
        mag, scale, pad_x, pad_y, w0, h0 = letterbox_rgb(
            self.root / mag_rel / self.split / f"{sid}.png", self.imgsz
        )
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        phase = None
        if self.dual:
            phase, *_ = letterbox_rgb(
                self.root / self.phase_subdir / self.split / f"{sid}.png", self.imgsz
            )
        labels = []
        lab_path = self.root / "labels" / self.split / f"{sid}.txt"
        if lab_path.exists():
            for line in lab_path.read_text().splitlines():
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls, x, y, w, h = int(p[0]), *map(float, p[1:])
                labels.append([cls, x, y, w, h])
        target = torch.zeros((0, 5), dtype=torch.float32)
        if labels:
            target = shift_yolo(torch.tensor(labels, dtype=torch.float32), pad_x, pad_y, nw, nh, self.imgsz)
        return {"mag": mag, "phase": phase, "target": target, "id": sid}


def collate(batch):
    mag = torch.stack([b["mag"] for b in batch], 0)
    phase = None
    if batch[0]["phase"] is not None:
        phase = torch.stack([b["phase"] for b in batch], 0)
    targets = [b["target"] for b in batch]
    ids = [b["id"] for b in batch]
    return mag, phase, targets, ids
