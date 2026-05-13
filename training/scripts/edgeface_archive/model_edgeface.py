"""
PyTorch model that mirrors the C engine's forward pass exactly.

Op order, shapes, and parameter layout match edgeface_engine.c so a
trained checkpoint can be exported as a flat tensor list (.bin) and
loaded by the engine without surgery.

NOTE: every learnable tensor created here ends up in the .bin in a
*specific order* — see export_bin.py for the contract. Don't reorder
sub-modules without updating the exporter.
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- low-level building blocks ---------------------------------------


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of an HWC-shaped tensor [B, H, W, C]."""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        # x: [B, H, W, C] or [B, N, C]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class LoRAMLP(nn.Module):
    """
    LoRA-decomposed MLP:
        x -> Lin(C,r) -> Lin(r,h) + bias -> GELU -> Lin(h,r) -> Lin(r,C) + bias

    Order of weights in the export list (per block):
        w0 = Lin(C, r).weight  shape [C, r] stored as [C, r] row-major
        w1 = Lin(r, h).weight  shape [r, h]
        b1 = Lin(r, h).bias    shape [h]
        w2 = Lin(h, r).weight  shape [h, r]
        w3 = Lin(r, C).weight  shape [r, C]
        b3 = Lin(r, C).bias    shape [C]
    """
    def __init__(self, c: int, rank: int, hidden: int):
        super().__init__()
        self.fc0 = nn.Linear(c, rank, bias=False)
        self.fc1 = nn.Linear(rank, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, rank, bias=False)
        self.fc3 = nn.Linear(rank, c, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ DW conv (NCHW) -> HWC -> LN -> LoRAMLP -> gamma * x + residual """
    def __init__(self, c: int, kernel: int, rank: int, hidden: int):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(c, c, kernel_size=kernel, padding=pad, groups=c, bias=True)
        self.ln = ChannelLayerNorm(c)
        self.mlp = LoRAMLP(c, rank, hidden)
        # ConvNeXt LayerScale: gamma_init=1e-6 keeps each block's residual
        # contribution tiny at start so activations don't explode through
        # 16+ residual blocks. Removed once and watched activations hit 1e5
        # by stage 3, collapsing embeddings to a single point.
        self.gamma = nn.Parameter(torch.full((c,), 1e-6))

    def forward(self, x):
        # x: [B, C, H, W]
        residual = x
        y = self.dw(x)
        y = y.permute(0, 2, 3, 1).contiguous()  # NHWC
        y = self.ln(y)
        y = self.mlp(y)
        y = y * self.gamma
        y = y.permute(0, 3, 1, 2).contiguous()  # NCHW
        return residual + y


class CascadedDWConv(nn.Module):
    """
    Cascaded depthwise conv used inside XCA:
      split channels into N groups, apply DW conv to first M groups
      where each conv input = (prev result) + (current split).
    Last (N-M) splits pass through unchanged.
    """
    def __init__(self, splits: Tuple[int, ...], n_convs: int, kernel: int = 3):
        super().__init__()
        self.splits = list(splits)
        self.n_convs = n_convs
        pad = kernel // 2
        self.convs = nn.ModuleList([
            nn.Conv2d(s, s, kernel_size=kernel, padding=pad, groups=s, bias=True)
            for s in self.splits[:n_convs]
        ])

    def forward(self, x):
        # x: [B, C, H, W]
        parts = list(torch.split(x, self.splits, dim=1))
        prev = None
        out = []
        for i, conv in enumerate(self.convs):
            inp = parts[i] if prev is None else (prev + parts[i])
            y = conv(inp)
            out.append(y)
            prev = y
        # remaining splits pass through unchanged
        out.extend(parts[len(self.convs):])
        return torch.cat(out, dim=1)


class XCABlock(nn.Module):
    """
    Cross-Covariance Attention block as used in EdgeFace.
    Components:
      - cascaded DW conv
      - optional positional embedding (1x1 conv on a learned [1,C,H,W] const)
      - LN -> LoRA QKV -> per-head L2-norm Q,K -> attn = (K^T Q) * temp -> softmax -> V@attn
      - LoRA out-proj -> gamma_xca * y + residual
      - LN -> LoRAMLP -> gamma * y + residual
    """
    def __init__(self, c: int, rank: int, hidden: int,
                 dw_splits: Tuple[int, ...], n_dw_convs: int,
                 dw_kernel: int = 3, n_heads: int = 4,
                 pos_embed: bool = False, pos_hw: int = 0):
        super().__init__()
        assert c % n_heads == 0, f"channels {c} must divide by n_heads {n_heads}"
        self.c, self.n_heads = c, n_heads
        self.dw = CascadedDWConv(dw_splits, n_dw_convs, dw_kernel)

        self.pos_embed = pos_embed
        if pos_embed:
            assert pos_hw > 0
            self.pos_const = nn.Parameter(torch.zeros(1, c, pos_hw, pos_hw))
            self.pos_conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

        self.ln_attn = ChannelLayerNorm(c)
        # QKV: LoRA(C -> r -> 3C)
        self.qkv_lora_a = nn.Linear(c, rank, bias=False)
        self.qkv_lora_b = nn.Linear(rank, 3 * c, bias=True)
        self.temperature = nn.Parameter(torch.full((n_heads,), 1.0))

        # output projection: LoRA(C -> r -> C)
        self.proj_lora_a = nn.Linear(c, rank, bias=False)
        self.proj_lora_b = nn.Linear(rank, c, bias=True)
        # LayerScale gamma_init=1e-6 — see ConvNeXtBlock comment.
        self.gamma_xca = nn.Parameter(torch.full((c,), 1e-6))

        self.ln_mlp = ChannelLayerNorm(c)
        self.mlp = LoRAMLP(c, rank, hidden)
        self.gamma = nn.Parameter(torch.full((c,), 1e-6))

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.dw(x)
        if self.pos_embed:
            pos = self.pos_conv(self.pos_const)        # [1, C, pos_h, pos_w]
            x = x + pos                                 # broadcast
        residual = x
        # to NHWC for LN + linear ops
        y = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = y.shape
        y = self.ln_attn(y)
        qkv = self.qkv_lora_b(self.qkv_lora_a(y))      # [B, H, W, 3C]
        qkv = qkv.reshape(B, H * W, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 4, 1)                # [3, B, n_heads, head_dim, HW]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # XCA: L2-norm along the spatial (HW) dim
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # attn: [B, n_heads, head_dim, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.temperature.view(1, self.n_heads, 1, 1)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                  # [B, n_heads, head_dim, HW]
        out = out.permute(0, 3, 1, 2).reshape(B, H, W, C)
        # output proj
        out = self.proj_lora_b(self.proj_lora_a(out))
        out = out * self.gamma_xca
        # back to NCHW for residual
        out = out.permute(0, 3, 1, 2).contiguous()
        y2 = residual + out

        # MLP
        residual2 = y2
        y3 = y2.permute(0, 2, 3, 1).contiguous()
        y3 = self.ln_mlp(y3)
        y3 = self.mlp(y3)
        y3 = y3 * self.gamma
        y3 = y3.permute(0, 3, 1, 2).contiguous()
        return residual2 + y3


class Downsample(nn.Module):
    """LN (channel-wise on HWC) -> Conv k=2 s=2 (NCHW)."""
    def __init__(self, c_in: int, c_out: int, kernel: int = 2, stride: int = 2):
        super().__init__()
        self.ln = ChannelLayerNorm(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, bias=True)

    def forward(self, x):
        # x: NCHW
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.ln(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.conv(y)
        return y


class FaceXModel(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch

        # stem
        self.stem = nn.Conv2d(3, arch.stem_out, kernel_size=arch.stem_kernel,
                              stride=arch.stem_stride, bias=True)
        self.stem_ln = ChannelLayerNorm(arch.stem_out)

        # stages
        self.stage_blocks = nn.ModuleList()  # parallel list of ModuleList per stage
        self.downsamples = nn.ModuleList()
        for s in arch.stages:
            blocks = nn.ModuleList()
            for _ in range(s.n_convnext):
                blocks.append(ConvNeXtBlock(s.channels, s.dw_kernel, s.rank, s.hidden))
            if s.has_xca:
                blocks.append(XCABlock(
                    s.channels, s.rank, s.hidden,
                    dw_splits=tuple(s.xca_splits), n_dw_convs=s.xca_n_convs,
                    dw_kernel=s.xca_dw_kernel, n_heads=s.xca_n_heads,
                    pos_embed=s.xca_pos_embed, pos_hw=s.xca_pos_hw,
                ))
            self.stage_blocks.append(blocks)
            if s.downsample_to:
                self.downsamples.append(Downsample(
                    s.channels, s.downsample_to,
                    kernel=s.downsample_kernel, stride=s.downsample_stride))
            else:
                self.downsamples.append(nn.Identity())

        # head
        last_c = arch.stages[-1].channels
        self.head_ln = ChannelLayerNorm(last_c)
        self.head_fc1 = nn.Linear(last_c, arch.head_rank, bias=False)
        self.head_fc2 = nn.Linear(arch.head_rank, arch.embedding_dim, bias=True)

    def forward(self, x):
        # x: [B, 3, 112, 112], values in [-1, 1]
        x = self.stem(x)                              # [B, C0, H/s, W/s]
        # LN on channel dim
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.stem_ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        for blocks, ds in zip(self.stage_blocks, self.downsamples):
            for blk in blocks:
                x = blk(x)
            x = ds(x)

        # head: GAP -> LN -> FC -> FC -> L2 norm
        x = x.mean(dim=(2, 3))                        # [B, C]
        x = self.head_ln(x)
        x = self.head_fc1(x)
        x = self.head_fc2(x)
        x = F.normalize(x, dim=-1)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------- ArcFace head (training only — not exported to .bin) -------------


class ArcFaceHead(nn.Module):
    """
    ArcFace large-margin cosine head.
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss".

    Uses the numerically-stable formulation from InsightFace's arcface_torch:
        cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
    where sin(θ) = sqrt(1 - cos²(θ)). This avoids `acos`, whose
    gradient blows up near ±1 — fatal when training in bf16.

    The "easy margin" guard switches off the margin when cos(θ) is in the
    region where adding margin would push θ past π (i.e. cos(θ) < cos(π - m)),
    avoiding the case where cos(θ + m) > cos(θ) for already-misclassified
    samples.
    """
    def __init__(self, emb_dim: int, n_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)   # below this, easy-margin kicks in
        self.mm = math.sin(math.pi - m) * m       # penalty used in easy-margin branch
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ArcFace's angle math is numerically delicate — force fp32 here even
        # when the surrounding training loop is in bf16 autocast.
        with torch.amp.autocast(emb.device.type, enabled=False):
            emb = emb.float()
            W = self.W.float()
            Wn = F.normalize(W, dim=-1)
            cos = F.linear(F.normalize(emb, dim=-1), Wn)        # [B, n_classes]
            # Pull the per-row target cosine
            idx = torch.arange(cos.size(0), device=cos.device)
            target_cos = cos[idx, labels]                       # [B]
            # cos(θ + m) via angle-addition (no acos!)
            sin = torch.sqrt(torch.clamp(1.0 - target_cos * target_cos, min=0.0))
            cos_m_target = target_cos * self.cos_m - sin * self.sin_m
            # Easy-margin guard: when cos(θ) is too small, fall back to (cos - mm)
            # so the loss stays monotone in θ.
            cos_m_target = torch.where(target_cos > self.threshold,
                                        cos_m_target,
                                        target_cos - self.mm)
            # Replace the target-class cosine in the [B, n_classes] tensor
            cos = cos.scatter(1, labels.unsqueeze(1), cos_m_target.unsqueeze(1))
            return cos * self.s
