"""Lightweight axis-separable dual-stream detector (torch).

Baseline A uses MagNet + loc/cls heads only.
Full MPFADet adds PhaseNet and occupancy-gated fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _pad(k):
    if isinstance(k, tuple):
        return tuple(x // 2 for x in k)
    return k // 2


def conv_bn_act(c1: int, c2: int, k=3, s: int = 1, g: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, _pad(k), groups=g, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


class AxisMix(nn.Module):
    """1xk time conv + kx1 freq conv."""

    def __init__(self, c: int, k_time: int = 7, k_freq: int = 3):
        super().__init__()
        self.time = conv_bn_act(c, c, k=(1, k_time), g=c)
        self.freq = conv_bn_act(c, c, k=(k_freq, 1), g=c)
        self.pw = conv_bn_act(c, c, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.time(x) + self.freq(x)) + x


class Stage(nn.Module):
    def __init__(self, c1: int, c2: int, n: int, stride: int, k_time: int, k_freq: int):
        super().__init__()
        self.down = conv_bn_act(c1, c2, k=3, s=stride)
        self.blocks = nn.Sequential(*[AxisMix(c2, k_time, k_freq) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class MagNet(nn.Module):
    def __init__(self, in_ch: int = 3, ch=(32, 64, 128, 256)):
        super().__init__()
        self.stem = conv_bn_act(in_ch, ch[0], k=3, s=2)
        self.s2 = Stage(ch[0], ch[1], n=2, stride=2, k_time=7, k_freq=3)
        self.s3 = Stage(ch[1], ch[2], n=2, stride=2, k_time=7, k_freq=3)
        self.s4 = Stage(ch[2], ch[3], n=2, stride=2, k_time=5, k_freq=3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p2 = self.stem(x)
        p3 = self.s2(p2)
        p4 = self.s3(p3)
        p5 = self.s4(p4)
        return [p2, p3, p4, p5]


class PhaseNet(nn.Module):
    def __init__(self, in_ch: int = 3, ch=(32, 64, 128, 256)):
        super().__init__()
        self.stem = conv_bn_act(in_ch, ch[0], k=3, s=2)
        self.s2 = Stage(ch[0], ch[1], n=2, stride=2, k_time=7, k_freq=3)
        self.s3 = Stage(ch[1], ch[2], n=2, stride=2, k_time=7, k_freq=3)
        self.s4 = Stage(ch[2], ch[3], n=2, stride=2, k_time=5, k_freq=3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p2 = self.stem(x)
        p3 = self.s2(p2)
        p4 = self.s3(p3)
        p5 = self.s4(p4)
        return [p2, p3, p4, p5]


class OccupancyGatedFusion(nn.Module):
    def __init__(self, c: int, alpha_init: float = 0.25, beta_init: float = 1.0):
        super().__init__()
        self.gate = nn.Conv2d(c, 1, 1)
        self.psi = nn.Conv2d(c, c, 1, bias=False)
        self.phi = nn.Conv2d(c, c, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, f_m: torch.Tensor, f_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gate(f_m))
        f_pg = f_p * g
        f_loc = f_m + self.alpha * self.psi(f_pg)
        f_cls = f_pg + self.beta * self.phi(f_m)
        return f_loc, f_cls, g


class DetectHead(nn.Module):
    """Decoupled HBB head: box (4) + obj (1) + cls (nc) per location, 3 strides."""

    def __init__(self, ch, nc: int = 9):
        super().__init__()
        self.nc = nc
        self.loc = nn.ModuleList()
        self.cls = nn.ModuleList()
        for c in ch:
            self.loc.append(
                nn.Sequential(conv_bn_act(c, c, 3), nn.Conv2d(c, 5, 1))  # l,t,r,b + obj
            )
            self.cls.append(
                nn.Sequential(conv_bn_act(c, c, 3), nn.Conv2d(c, nc, 1))
            )

    def forward(self, loc_feats, cls_feats):
        out = []
        for i, (fl, fc) in enumerate(zip(loc_feats, cls_feats)):
            out.append(torch.cat([self.loc[i](fl), self.cls[i](fc)], dim=1))
        return out


class MPFADet(nn.Module):
    def __init__(self, nc: int = 9, dual: bool = True, ch=(32, 64, 128, 256), p2: bool = False):
        super().__init__()
        self.dual = dual
        self.p2 = p2
        self.mag = MagNet(3, ch)
        self.phase = PhaseNet(3, ch) if dual else None
        head_ch = ch if p2 else ch[1:]
        self.fuse = nn.ModuleList([OccupancyGatedFusion(c) for c in head_ch]) if dual else None
        self.head = DetectHead(head_ch, nc)
        # stem is stride 2; s2/s3/s4 add stride 2 each → 2/4/8/16 if p2 else 4/8/16
        self.strides = (2, 4, 8, 16) if p2 else (4, 8, 16)

    def _levels(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return feats if self.p2 else feats[1:]

    def forward(self, mag: torch.Tensor, phase: torch.Tensor | None = None):
        fm = self._levels(self.mag(mag))
        if self.dual:
            if phase is None:
                raise ValueError("dual model needs phase input")
            fp = self._levels(self.phase(phase))
            loc, cls = [], []
            gates = []
            for i, fuse in enumerate(self.fuse):
                fl, fc, g = fuse(fm[i], fp[i])
                loc.append(fl)
                cls.append(fc)
                gates.append(g)
            preds = self.head(loc, cls)
            return preds, gates
        preds = self.head(fm, fm)
        return preds, None
