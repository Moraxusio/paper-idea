"""Locked TF geometry from Step 1."""

T_MAX = 5.0e7
F_MAX = 5.0e4
FS = 50_000.0
PNG_W = 875
PNG_H = 656
DURATION_S = 5.0

CLASS_TO_ID = {
    "2FSK": 0,
    "4FSK": 1,
    "8-Tone": 2,
    "16-Tone": 3,
    "GMSK": 4,
    "FM": 5,
    "AM-DSB": 6,
    "Morse": 7,
    "PSK": 8,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def box_xyxy(box: dict, w: int = PNG_W, h: int = PNG_H) -> tuple[int, int, int, int]:
    x0 = box["DateTimeStart"] / T_MAX * w
    x1 = box["DateTimeEnd"] / T_MAX * w
    y0 = (1.0 - box["FreqU"] / F_MAX) * h
    y1 = (1.0 - box["FreqD"] / F_MAX) * h
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return (
        max(0, min(w - 1, int(xa))),
        max(0, min(h - 1, int(ya))),
        max(1, min(w, int(xb + 0.999))),
        max(1, min(h, int(yb + 0.999))),
    )


def box_yolo(box: dict) -> tuple[int, float, float, float, float]:
    cls = CLASS_TO_ID[box["Content"]]
    x = (box["DateTimeStart"] + box["DateTimeEnd"]) / 2.0 / T_MAX
    w = (box["DateTimeEnd"] - box["DateTimeStart"]) / T_MAX
    y = 1.0 - (box["FreqD"] + box["FreqU"]) / 2.0 / F_MAX
    h = (box["FreqU"] - box["FreqD"]) / F_MAX
    return cls, x, y, w, h
