from __future__ import annotations

import numpy as np


def hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1))


def stft_complex(
    x: np.ndarray,
    n_fft: int = 1024,
    hop: int = 256,
    fftshift: bool = False,
) -> np.ndarray:
    """Return STFT with shape (n_freq, n_time), freq axis 0 .. fs (or -fs/2 .. fs/2 if shifted)."""
    win = hann(n_fft).astype(np.float32)
    n = x.shape[0]
    if n < n_fft:
        raise ValueError("signal shorter than n_fft")
    n_frames = 1 + (n - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = x[idx] * win[None, :]
    spec = np.fft.fft(frames, n=n_fft, axis=1)
    if fftshift:
        spec = np.fft.fftshift(spec, axes=1)
    return spec.T.astype(np.complex64)


def resize_tf(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize for 2D arrays, (F, T) or (H, W) -> (out_h, out_w)."""
    src_h, src_w = arr.shape[:2]
    ys = (np.linspace(0, src_h - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, src_w - 1, out_w)).astype(np.int32)
    return arr[ys[:, None], xs[None, :]]


def box_mean(arr: np.ndarray, kt: int, kf: int) -> np.ndarray:
    """Separable box mean. arr is (F, T). kf along freq, kt along time. Odd sizes."""
    assert kt % 2 == 1 and kf % 2 == 1
    pad_f = kf // 2
    pad_t = kt // 2
    x = np.pad(arr, ((pad_f, pad_f), (pad_t, pad_t)), mode="edge")
    c = np.cumsum(x, axis=0)
    c = np.concatenate([np.zeros((1,) + c.shape[1:], dtype=c.dtype), c], axis=0)
    vf = c[kf:, :] - c[:-kf, :]
    c2 = np.cumsum(vf, axis=1)
    c2 = np.concatenate([np.zeros((c2.shape[0], 1), dtype=c2.dtype), c2], axis=1)
    out = c2[:, kt:] - c2[:, :-kt]
    return out / float(kt * kf)
