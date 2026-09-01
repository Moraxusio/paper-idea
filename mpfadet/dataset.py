from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


class MagPhaseDataset(Dataset):
    def __init__(self, root: Path, split: str, imgsz: int = 512, dual: bool = False):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.dual = dual
        mag_dir = self.root / "images" / split
        self.ids = sorted(p.stem for p in mag_dir.glob("*.png"))
        if not self.ids:
            raise FileNotFoundError(f"no images in {mag_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        mag = load_rgb(self.root / "images" / self.split / f"{sid}.png", self.imgsz)
        phase = None
        if self.dual:
            phase = load_rgb(self.root / "image" / self.split / f"{sid}.png", self.imgsz)
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
            target = torch.tensor(labels, dtype=torch.float32)
        return {"mag": mag, "phase": phase, "target": target, "id": sid}


def collate(batch):
    mag = torch.stack([b["mag"] for b in batch], 0)
    phase = None
    if batch[0]["phase"] is not None:
        phase = torch.stack([b["phase"] for b in batch], 0)
    targets = [b["target"] for b in batch]
    ids = [b["id"] for b in batch]
    return mag, phase, targets, ids
