"""
Tiny YuNet-style face detector. Anchor-free FCOS-like heads with 5-point
landmark regression.

Backbone:
    Stem  3x3 s=2     3 → 16
    s1    IB×2 s=2    16 → 24
    s2    IB×2 s=2    24 → 32    → scale 1 (1/8)
    s3    IB×2 s=2    32 → 48    → scale 2 (1/16)
    s4    IB×2 s=2    48 → 64    → scale 3 (1/32)

Heads (shared trunk per scale):
    trunk: Conv1x1 → 64 + Conv3x3 dw → 64
    cls:   Conv1x1 → 1   (face/bg logit, sigmoid at inference)
    box:   Conv1x1 → 4   (l, t, r, b distances in stride units)
    kps:   Conv1x1 → 10  (5 keypoints, normalized offsets from anchor center)

Anchor centers: each cell at (cx+0.5, cy+0.5) * stride.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class InvBottleneck(nn.Module):
    """1x1 expand → 3x3 dw → 1x1 project + residual if stride=1 and matched channels."""
    def __init__(self, ci, co, s=1, expand=4):
        super().__init__()
        hidden = max(8, ci * expand)
        self.expand = ConvBN(ci, hidden, 1)
        self.dw = ConvBN(hidden, hidden, 3, s, 1, g=hidden)
        self.project = ConvBN(hidden, co, 1, act=False)
        self.use_res = (s == 1 and ci == co)

    def forward(self, x):
        y = self.project(self.dw(self.expand(x)))
        if self.use_res: y = y + x
        return y


class Head(nn.Module):
    """Per-scale detection head. Predicts cls (1), box (4), kps (10)."""
    def __init__(self, ci, h=32):
        super().__init__()
        self.trunk = nn.Sequential(
            ConvBN(ci, h, 1),
            ConvBN(h, h, 3, 1, 1, g=h),   # dw
            ConvBN(h, h, 1),
        )
        self.cls = nn.Conv2d(h, 1, 1)
        self.box = nn.Conv2d(h, 4, 1)
        self.kps = nn.Conv2d(h, 10, 1)
        # Bias the cls head toward negatives to stabilize early training (RetinaFace trick).
        nn.init.constant_(self.cls.bias, -4.59)   # logit(0.01)
        nn.init.normal_(self.box.weight, std=0.01); nn.init.zeros_(self.box.bias)
        nn.init.normal_(self.kps.weight, std=0.01); nn.init.zeros_(self.kps.bias)

    def forward(self, x):
        f = self.trunk(x)
        return self.cls(f), self.box(f), self.kps(f)


class FaceDetector(nn.Module):
    """Output strides: 8, 16, 32. Trained on 320×320 by default."""
    STRIDES = (8, 16, 32)
    INPUT_SIZE = 320

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(3, 16, 3, 2, 1)
        # s1 (1/4): 16 → 24
        self.s1 = nn.Sequential(
            InvBottleneck(16, 24, s=2, expand=2),
            InvBottleneck(24, 24, s=1, expand=2),
        )
        # s2 (1/8): 24 → 32  → P3
        self.s2 = nn.Sequential(
            InvBottleneck(24, 32, s=2, expand=3),
            InvBottleneck(32, 32, s=1, expand=3),
        )
        # s3 (1/16): 32 → 48 → P4
        self.s3 = nn.Sequential(
            InvBottleneck(32, 48, s=2, expand=3),
            InvBottleneck(48, 48, s=1, expand=3),
        )
        # s4 (1/32): 48 → 64 → P5
        self.s4 = nn.Sequential(
            InvBottleneck(48, 64, s=2, expand=3),
            InvBottleneck(64, 64, s=1, expand=3),
        )
        self.head_p3 = Head(32)
        self.head_p4 = Head(48)
        self.head_p5 = Head(64)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        p3 = self.s2(x)
        p4 = self.s3(p3)
        p5 = self.s4(p4)
        c3, b3, k3 = self.head_p3(p3)
        c4, b4, k4 = self.head_p4(p4)
        c5, b5, k5 = self.head_p5(p5)
        return (c3, b3, k3), (c4, b4, k4), (c5, b5, k5)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def make_anchors(input_size: int, strides=FaceDetector.STRIDES, device="cpu"):
    """Return [N, 2] tensor of anchor centers in pixel coords, plus [N] strides."""
    centers = []
    strides_per = []
    for s in strides:
        n = input_size // s
        ys, xs = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device),
                                 indexing="ij")
        cxy = torch.stack([xs.flatten(), ys.flatten()], -1).float() + 0.5
        cxy = cxy * s
        centers.append(cxy)
        strides_per.append(torch.full((cxy.shape[0],), s, dtype=torch.float32, device=device))
    return torch.cat(centers, 0), torch.cat(strides_per, 0)


if __name__ == "__main__":
    m = FaceDetector().eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}  ~{n*4/1024:.1f} KB FP32  ~{n/1024:.1f} KB INT8")
    x = torch.zeros(1, 3, 320, 320)
    outs = m(x)
    for i, (c, b, k) in enumerate(outs):
        print(f"  P{i+3}: cls={tuple(c.shape)}  box={tuple(b.shape)}  kps={tuple(k.shape)}")
    a, s = make_anchors(320)
    print(f"anchors: {a.shape[0]}  total cells across scales")
