from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np


def read_iq_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        fs = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if nch != 2 or sw != 2:
        raise ValueError(f"unexpected wav format ch={nch} sw={sw} in {path}")
    x = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    x /= 32768.0
    return x, fs


def parse_dr(path: Path) -> list[dict]:
    boxes = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            boxes.append(json.loads(line))
    return boxes


def list_sample_ids(raw_dir: Path) -> list[str]:
    ids = []
    for png in sorted(Path(raw_dir).glob("*_spectrogram.png")):
        ids.append(png.name[: -len("_spectrogram.png")])
    return ids
