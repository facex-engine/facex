"""Tiny binary tongue classifier. ~50K params, 112x112 RGB → 2 classes."""

import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, ci, co, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(co)
        self.act = nn.PReLU(co) if act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act is not None: x = self.act(x)
        return x


class InvBN(nn.Module):
    def __init__(self, ci, co, s=1, t=2):
        super().__init__()
        h = max(8, ci * t)
        self.use_res = (s == 1 and ci == co)
        self.expand = ConvBN(ci, h, 1)
        self.dw = ConvBN(h, h, 3, s, 1, g=h)
        self.project = ConvBN(h, co, 1, act=False)

    def forward(self, x):
        y = self.project(self.dw(self.expand(x)))
        return y + x if self.use_res else y


class SmileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBN(3, 16, 3, 2, 1)
        # 112 → 56
        self.s1 = nn.Sequential(InvBN(16, 24, 2, 2), InvBN(24, 24, 1, 2))   # 28
        self.s2 = nn.Sequential(InvBN(24, 32, 2, 3), InvBN(32, 32, 1, 3))   # 14
        self.s3 = nn.Sequential(InvBN(32, 48, 2, 3), InvBN(48, 48, 1, 3))   # 7
        self.head_conv = ConvBN(48, 64, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.stem(x); x = self.s1(x); x = self.s2(x); x = self.s3(x)
        x = self.head_conv(x); x = self.pool(x).flatten(1)
        return self.fc(x)

    def num_params(self): return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = SmileNet().eval()
    print(f"params: {m.num_params():,}")
    x = torch.zeros(1, 3, 112, 112)
    y = m(x)
    print(f"input {tuple(x.shape)} → output {tuple(y.shape)}")
