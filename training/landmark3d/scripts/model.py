"""
3D face-landmark regressor — distilled from MediaPipe FaceMesh.

Input: [B, 3, 112, 112] face crop, values in [-1, 1].
Output: [B, 478, 3] landmarks: x, y in [0, 1] (crop-local), z roughly [-0.5, 0.5].

Architecture: MobileFaceNet-style backbone (~2M params, ~3MB INT8),
GDConv + FC head producing 478*3=1434 floats.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_PTS = 478


class ConvBN(nn.Module):
    def __init__(self, ci, co, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.PReLU(co) if act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act is not None: x = self.act(x)
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
        if self.use_residual: y = y + x
        return y


class Landmark3DNet(nn.Module):
    NUM_PTS = NUM_PTS
    # Compact: matches our 98-pt landmark model but with 3D output.
    STAGES = [
        # (expand, channels, n, stride)
        (1, 32,  1, 1),
        (4, 32,  2, 1),
        (4, 64,  2, 2),
        (4, 64,  3, 1),
        (4, 96,  2, 2),
        (4, 96,  3, 1),
        (4, 128, 2, 2),
        (4, 128, 2, 1),
    ]

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(3, 32, 3, s=2, p=1)         # 112 -> 56
        layers = []
        ci = 32
        for t, c, n, s in self.STAGES:
            for j in range(n):
                stride = s if j == 0 else 1
                layers.append(InvBottleneck(ci, c, stride, t))
                ci = c
        self.blocks = nn.Sequential(*layers)
        self.final = ConvBN(ci, 256, 1)
        self.gdc = ConvBN(256, 256, 7, s=1, p=0, g=256, act=False)
        self.fc = nn.Linear(256, NUM_PTS * 3)
        # Init bias: centered xy=0.5, z=0
        nn.init.zeros_(self.fc.weight)
        bias = torch.zeros(NUM_PTS * 3)
        bias.view(NUM_PTS, 3)[:, :2] = 0.5
        with torch.no_grad():
            self.fc.bias.copy_(bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        x = self.gdc(x).flatten(1)
        return self.fc(x).view(-1, NUM_PTS, 3)

    def num_params(self): return sum(p.numel() for p in self.parameters())


# -------- Distillation loss --------


class DistillLoss(nn.Module):
    """L1 on xy + L1 on z (separately weighted)."""
    def __init__(self, w_xy: float = 1.0, w_z: float = 0.5):
        super().__init__()
        self.w_xy = w_xy; self.w_z = w_z

    def forward(self, pred, target):
        # both [B, 478, 3]. No .item() — avoid GPU↔CPU sync inside training loop.
        loss_xy = (pred[..., :2] - target[..., :2]).abs().mean() * 112.0
        loss_z  = (pred[..., 2]  - target[..., 2]).abs().mean() * 112.0
        return self.w_xy * loss_xy + self.w_z * loss_z
