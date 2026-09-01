from __future__ import annotations

import numpy as np

from .geometry import PNG_H, PNG_W
from .stft import box_mean, resize_tf, stft_complex


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def build_complex(iq: np.ndarray, swap: bool, conj: bool) -> np.ndarray:
    i = iq[:, 0]
    q = iq[:, 1]
    if swap:
        i, q = q, i
    x = i + 1j * q
    if conj:
        x = np.conj(x)
    return x


def compute_fields(x: np.ndarray, n_fft: int, hop: int, fs: float, fftshift: bool) -> dict:
    X = stft_complex(x, n_fft=n_fft, hop=hop, fftshift=fftshift)
    mag = np.abs(X)
    logmag = np.log(mag + 1e-8)
    phi = np.angle(X)
    unit = np.exp(1j * phi)
    coh = np.abs(box_mean(unit, kt=5, kf=5)).astype(np.float32)

    prev = X[:, :-1]
    curr = X[:, 1:]
    dphi = np.angle(curr * np.conj(prev))
    dphi_pad = np.concatenate([dphi[:, :1], dphi], axis=1)
    inst_f = dphi_pad / (2.0 * np.pi) * (fs / hop)

    slow = box_mean(dphi_pad.astype(np.float64), kt=9, kf=1)
    resid = np.sin(dphi_pad - slow)

    return {
        "X": X,
        "mag": mag.astype(np.float32),
        "logmag": logmag.astype(np.float32),
        "if": inst_f.astype(np.float32),
        "coh": coh,
        "resid": resid.astype(np.float32),
    }


def fields_to_pem(
    fields: dict,
    out_h: int = PNG_H,
    out_w: int = PNG_W,
    flip_freq: bool = True,
    gate_alpha: float = 6.0,
    gate_percentile: float = 55.0,
    if_scale_hz: float | None = None,
    hop: int = 256,
    n_fft: int = 1024,
) -> np.ndarray:
    logmag = fields["logmag"]
    inst_f = fields["if"]
    coh = fields["coh"]
    resid = fields["resid"]
    if flip_freq:
        logmag = logmag[::-1]
        inst_f = inst_f[::-1]
        coh = coh[::-1]
        resid = resid[::-1]

    if if_scale_hz is None:
        bin_hz = 50000.0 / n_fft
        if_scale_hz = 0.5 * bin_hz * (n_fft / hop)  # ~ 2 bins of IF in Hz units of hop-normalized
        if_scale_hz = max(float(np.percentile(np.abs(inst_f), 90)), 50.0)

    r = np.clip(inst_f / if_scale_hz, -1.0, 1.0) * 0.5 + 0.5
    g = np.clip(coh, 0.0, 1.0)
    b = np.clip(resid * 0.5 + 0.5, 0.0, 1.0)

    tau = np.percentile(logmag, gate_percentile)
    gate = sigmoid(gate_alpha * (logmag - tau)).astype(np.float32)
    pem_f = np.stack([r, g, b], axis=-1) * gate[..., None]
    pem_f = np.clip(pem_f, 0.0, 1.0)
    pem = resize_tf(pem_f, out_h, out_w)
    return (pem * 255.0 + 0.5).astype(np.uint8)


def logmag_image(logmag: np.ndarray, out_h: int, out_w: int, flip_freq: bool = True) -> np.ndarray:
    x = logmag[::-1] if flip_freq else logmag
    x = resize_tf(x, out_h, out_w)
    lo, hi = np.percentile(x, [5, 99])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (x * 255.0 + 0.5).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)
