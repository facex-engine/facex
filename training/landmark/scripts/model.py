"""
Small face-landmark regressor. 98 (x,y) outputs in [0,1] normalized to a
112x112 crop. Architecture: stem + 5 MobileNet-style bottlenecks + GAP +
FC. Compact enough to run in real time in WASM (~600K params, ~600KB INT8).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, ci, co, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.PReLU(co) if act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x


class InvBottleneck(nn.Module):
    def __init__(self, ci, co, stride, expand):
        super().__init__()
        hidden = ci * expand
        self.use_residual = (stride == 1 and ci == co)
        self.expand  = ConvBN(ci, hidden, 1)
        self.dw      = ConvBN(hidden, hidden, 3, s=stride, p=1, g=hidden)
        self.project = ConvBN(hidden, co, 1, act=False)

    def forward(self, x):
        y = self.project(self.dw(self.expand(x)))
        if self.use_residual:
            y = y + x
        return y


class LandmarkNet(nn.Module):
    """
    Input:  [B, 3, 112, 112], values in [-1, 1]
    Output: [B, 196]  (98 landmarks; (x0,y0,x1,y1,...) in [0,1])
    """
    DEFAULT_STAGES = [
        # (expand, channels, n, stride)   spatial after stage (stem made 56)
        (1, 32,  1, 1),  # 56
        (4, 32,  2, 1),  # 56
        (4, 64,  2, 2),  # 28
        (4, 64,  3, 1),  # 28
        (4, 96,  2, 2),  # 14
        (4, 96,  3, 1),  # 14
        (4, 128, 2, 2),  # 7
        (4, 128, 2, 1),  # 7
    ]
    NUM_PTS = 98

    def __init__(self, stages=None):
        super().__init__()
        stages = stages or self.DEFAULT_STAGES
        self.stem = ConvBN(3, 32, 3, s=2, p=1)         # 112 -> 56
        layers = []
        ci = 32
        for t, c, n, s in stages:
            for j in range(n):
                stride = s if j == 0 else 1
                layers.append(InvBottleneck(ci, c, stride, t))
                ci = c
        self.blocks = nn.Sequential(*layers)
        self.final = ConvBN(ci, 256, 1)
        # GDConv: depthwise 7x7 -> 1x1
        self.gdc = ConvBN(256, 256, 7, s=1, p=0, g=256, act=False)
        self.fc = nn.Linear(256, self.NUM_PTS * 2)
        # init final FC bias to "centered face" (all coords ~0.5)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        x = self.gdc(x)        # [B, 256, 1, 1]
        x = x.flatten(1)
        x = self.fc(x)         # [B, 196]
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---- Wing loss (Feng et al. 2018) — standard for face landmark training ----


class WingLoss(nn.Module):
    def __init__(self, w: float = 10.0, eps: float = 2.0):
        super().__init__()
        self.w = w
        self.eps = eps
        self.C = w - w * math.log(1 + w / eps)

    def forward(self, pred, target):
        # pred, target: [B, 196]; coordinates in [0,1]. Scale to 112-pixel
        # units so the wing-loss "knee" at `w` lives in pixel space.
        diff = (pred - target) * 112.0
        absd = diff.abs()
        below = absd < self.w
        loss = torch.where(
            below,
            self.w * torch.log(1 + absd / self.eps),
            absd - self.C,
        )
        return loss.mean()
