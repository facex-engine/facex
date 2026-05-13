"""Compact anti-spoof binary classifier. ~1M params, MobileFaceNet-style."""

import torch
import torch.nn as nn


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


class AntiSpoofNet(nn.Module):
    STAGES = [
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
        self.stem = ConvBN(3, 32, 3, s=2, p=1)
        layers = []; ci = 32
        for t, c, n, s in self.STAGES:
            for j in range(n):
                stride = s if j == 0 else 1
                layers.append(InvBottleneck(ci, c, stride, t))
                ci = c
        self.blocks = nn.Sequential(*layers)
        self.final = ConvBN(ci, 256, 1)
        self.gdc = ConvBN(256, 256, 7, s=1, p=0, g=256, act=False)
        self.fc = nn.Linear(256, 2)             # 2 classes: spoof, live

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        x = self.gdc(x).flatten(1)
        return self.fc(x)

    def num_params(self): return sum(p.numel() for p in self.parameters())
