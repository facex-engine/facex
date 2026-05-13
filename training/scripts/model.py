"""
MobileFaceNet PyTorch implementation.

Reference: Chen et al. 2018, "MobileFaceNets: Efficient CNNs for Accurate
Real-Time Face Verification on Mobile Devices."

The op order, parameter naming, and weight ordering are designed so a
trained checkpoint can be exported to a flat .bin and consumed by a small
C engine (BN-fused into the conv weights at export time).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- building blocks ------------------------------------------------


class ConvBN(nn.Module):
    """Conv2d + BN, no activation (used for linear bottleneck projection)."""
    def __init__(self, c_in: int, c_out: int, kernel: int = 1, stride: int = 1,
                 padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNPReLU(nn.Module):
    """Conv2d + BN + PReLU (per-channel)."""
    def __init__(self, c_in: int, c_out: int, kernel: int = 1, stride: int = 1,
                 padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.PReLU(c_out)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """
    MobileNet-V2 style inverted residual:
        1x1 expand -> BN -> PReLU
        DW 3x3 (s=stride) -> BN -> PReLU
        1x1 project -> BN (no activation — "linear bottleneck")
        + residual (only when stride==1 and c_in==c_out)
    """
    def __init__(self, c_in: int, c_out: int, stride: int, expand: int):
        super().__init__()
        hidden = c_in * expand
        self.use_residual = (stride == 1 and c_in == c_out)
        self.expand = ConvBNPReLU(c_in, hidden, kernel=1)
        self.dw = ConvBNPReLU(hidden, hidden, kernel=3, stride=stride,
                              padding=1, groups=hidden)
        self.project = ConvBN(hidden, c_out, kernel=1)

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.project(y)
        if self.use_residual:
            y = y + x
        return y


# ---------- MobileFaceNet --------------------------------------------------


class MobileFaceNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        c_stem = arch.stem_channels
        c_block = arch.block_channels
        c_final = arch.final_channels
        emb_dim = arch.embedding_dim

        # 112x112 -> 56x56
        self.stem = ConvBNPReLU(3, c_stem, kernel=3, stride=2, padding=1)
        # 56x56 (DW conv after stem)
        self.dw_stem = ConvBNPReLU(c_stem, c_stem, kernel=3, stride=1,
                                    padding=1, groups=c_stem)

        # bottleneck stages
        stages = []
        c_in = c_stem
        for i, s in enumerate(arch.stages):
            c_out = c_stem if i == 0 else c_block
            for j in range(s.n):
                stride = s.s if j == 0 else 1
                ci = c_in if j == 0 else c_out
                stages.append(InvertedResidual(ci, c_out, stride, expand=s.t))
            c_in = c_out
        self.stages = nn.Sequential(*stages)

        # 7x7 -> 7x7, expand to final_channels
        self.final_conv = ConvBNPReLU(c_in, c_final, kernel=1)
        # GDConv: DW conv with kernel=feature_size, no padding -> 1x1 spatial
        self.gdconv = ConvBN(c_final, c_final, kernel=arch.gdc_kernel,
                              stride=1, padding=0, groups=c_final)
        # 1x1 conv to embedding dim, then final BN. No activation, no L2 norm
        # in forward — InsightFace eval normalizes externally before cosine.
        # We do L2 normalize at the end so embeddings are unit vectors (the
        # ArcFace head expects this contract).
        self.embed_proj = ConvBN(c_final, emb_dim, kernel=1)

    def forward(self, x):
        # x: [B, 3, 112, 112], values in [-1, 1]
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        x = self.gdconv(x)        # [B, c_final, 1, 1]
        x = self.embed_proj(x)    # [B, emb_dim, 1, 1]
        x = x.flatten(1)          # [B, emb_dim]
        x = F.normalize(x, dim=-1)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------- ArcFace head (training only — not exported to .bin) ------------


class ArcFaceHead(nn.Module):
    """
    ArcFace large-margin head. Uses the numerically-stable angle-addition
    formulation (no acos), and forces fp32 even under bf16 autocast.
    """
    def __init__(self, emb_dim: int, n_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(emb.device.type, enabled=False):
            emb = emb.float()
            W = self.W.float()
            Wn = F.normalize(W, dim=-1)
            cos = F.linear(F.normalize(emb, dim=-1), Wn)
            idx = torch.arange(cos.size(0), device=cos.device)
            target_cos = cos[idx, labels]
            sin = torch.sqrt(torch.clamp(1.0 - target_cos * target_cos, min=0.0))
            cos_m_target = target_cos * self.cos_m - sin * self.sin_m
            cos_m_target = torch.where(target_cos > self.threshold,
                                        cos_m_target,
                                        target_cos - self.mm)
            cos = cos.scatter(1, labels.unsqueeze(1), cos_m_target.unsqueeze(1))
            return cos * self.s


# Public alias for train.py
FaceXModel = MobileFaceNet
