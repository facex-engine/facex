"""
FaceX .bin v2 format — parametric, self-describing.

File layout (all integers little-endian):
    magic         : 4 bytes  = b"EFX2"
    version       : u32      = 2
    binary header : ARCH_HDR_SIZE bytes (struct below)
    json_len      : u32      (debug copy of the arch as JSON, used by Python)
    json          : json_len bytes (UTF-8)
    n_tensors     : u32
    tensor records: per-tensor [u32 size][raw FP32 bytes]

The C engine reads ONLY the binary header (skips the JSON via json_len).
The JSON exists so Python tools and humans can inspect a .bin without
needing to know the binary schema.

Binary header schema:
    u8  hdr_version (=1)
    u8  n_stages            (1..MAX_STAGES=6)
    u8  stem_in_c (=3)
    u8  stem_kernel
    u8  stem_stride
    u8  pad
    u16 stem_out
    u16 head_rank
    u16 embedding_dim

    per stage (STAGE_REC_SIZE = 32 bytes, MAX_STAGES of them):
        u16 channels
        u8  n_convnext
        u8  dw_kernel
        u16 rank
        u16 hidden
        u8  has_xca
        u8  has_pos_embed
        u8  pos_hw
        u8  n_dw_splits
        u8  n_dw_convs
        u8  xca_dw_kernel
        u8  xca_n_heads
        u8  pad1
        u16 dw_splits[4]   (zero-padded; max 4 splits supported)
        u8  has_downsample
        u8  downsample_kernel
        u8  downsample_stride
        u8  pad2
        u16 downsample_to
        u16 pad3

Total binary header: 12 + 6*32 = 204 bytes.

Tensor order is fixed and produced by `tensor_layout(arch)` below.
"""

import struct as _struct

MAX_STAGES = 6
STAGE_REC_SIZE = 32
ARCH_HDR_SIZE = 12 + MAX_STAGES * STAGE_REC_SIZE  # 204

import json
import struct
from dataclasses import asdict
from typing import List, Tuple


# ---------- Tensor layout ---------------------------------------------------
# A "tensor" is one nn.Parameter. The order below is the contract between
# Python exporter and C engine. Add new tensors at the *end* if extending.
#
# Naming convention used in the layout strings:
#   stem.conv.{w,b}          : stem 4x4 conv
#   stem.ln.{w,b}            : stem channel LN
#   stage{i}.block{j}.dw.{w,b}
#   stage{i}.block{j}.ln.{w,b}
#   stage{i}.block{j}.mlp.{fc0,fc1,fc2,fc3}.{w,b}
#   stage{i}.block{j}.gamma
#   stage{i}.xca.dw{k}.{w,b}    (k = 0..n_dw_convs-1)
#   stage{i}.xca.pos_const     (only stage with pos_embed)
#   stage{i}.xca.pos_conv.{w,b}
#   stage{i}.xca.ln_attn.{w,b}
#   stage{i}.xca.qkv.{a,b}.{w,b}    (a = LoRA-A no bias; b = LoRA-B with bias)
#   stage{i}.xca.temperature
#   stage{i}.xca.proj.{a,b}.{w,b}
#   stage{i}.xca.gamma_xca
#   stage{i}.xca.ln_mlp.{w,b}
#   stage{i}.xca.mlp.{fc0..fc3}.{w,b}
#   stage{i}.xca.gamma
#   stage{i}.downsample.ln.{w,b}
#   stage{i}.downsample.conv.{w,b}
#   head.ln.{w,b}
#   head.fc1.w
#   head.fc2.{w,b}


def tensor_layout(arch) -> List[str]:
    names: List[str] = []
    names.append("stem.conv.w");   names.append("stem.conv.b")
    names.append("stem.ln.w");     names.append("stem.ln.b")
    for i, s in enumerate(arch.stages):
        for j in range(s.n_convnext):
            p = f"stage{i}.block{j}"
            names += [f"{p}.dw.w",  f"{p}.dw.b",
                      f"{p}.ln.w",  f"{p}.ln.b",
                      f"{p}.mlp.fc0.w",
                      f"{p}.mlp.fc1.w", f"{p}.mlp.fc1.b",
                      f"{p}.mlp.fc2.w",
                      f"{p}.mlp.fc3.w", f"{p}.mlp.fc3.b",
                      f"{p}.gamma"]
        if s.has_xca:
            xp = f"stage{i}.xca"
            for k in range(s.xca_n_convs):
                names += [f"{xp}.dw{k}.w", f"{xp}.dw{k}.b"]
            if s.xca_pos_embed:
                names += [f"{xp}.pos_const",
                          f"{xp}.pos_conv.w", f"{xp}.pos_conv.b"]
            names += [f"{xp}.ln_attn.w", f"{xp}.ln_attn.b",
                      f"{xp}.qkv.a.w",
                      f"{xp}.qkv.b.w", f"{xp}.qkv.b.b",
                      f"{xp}.temperature",
                      f"{xp}.proj.a.w",
                      f"{xp}.proj.b.w", f"{xp}.proj.b.b",
                      f"{xp}.gamma_xca",
                      f"{xp}.ln_mlp.w", f"{xp}.ln_mlp.b",
                      f"{xp}.mlp.fc0.w",
                      f"{xp}.mlp.fc1.w", f"{xp}.mlp.fc1.b",
                      f"{xp}.mlp.fc2.w",
                      f"{xp}.mlp.fc3.w", f"{xp}.mlp.fc3.b",
                      f"{xp}.gamma"]
        if s.downsample_to:
            names += [f"stage{i}.downsample.ln.w", f"stage{i}.downsample.ln.b",
                      f"stage{i}.downsample.conv.w", f"stage{i}.downsample.conv.b"]
    names += ["head.ln.w", "head.ln.b",
              "head.fc1.w",
              "head.fc2.w", "head.fc2.b"]
    return names


def arch_to_dict(arch) -> dict:
    """Serializable arch description (matches JSON in the .bin header)."""
    return {
        "name": arch.name,
        "stem_out": arch.stem_out,
        "stem_kernel": arch.stem_kernel,
        "stem_stride": arch.stem_stride,
        "head_rank": arch.head_rank,
        "embedding_dim": arch.embedding_dim,
        "stages": [
            {
                "channels": s.channels, "n_convnext": s.n_convnext,
                "dw_kernel": s.dw_kernel, "rank": s.rank, "hidden": s.hidden,
                "has_xca": s.has_xca, "xca_splits": list(s.xca_splits),
                "xca_n_convs": s.xca_n_convs, "xca_dw_kernel": s.xca_dw_kernel,
                "xca_n_heads": s.xca_n_heads,
                "xca_pos_embed": s.xca_pos_embed, "xca_pos_hw": s.xca_pos_hw,
                "downsample_to": s.downsample_to,
                "downsample_kernel": s.downsample_kernel,
                "downsample_stride": s.downsample_stride,
            } for s in arch.stages
        ],
    }


_HDR_FIXED = struct.Struct("<BBBBBBHHH")  # 12 bytes
_STAGE_REC = struct.Struct("<HBBHHBBBBBBBBHHHHBBBBHH")  # 32 bytes


def _pack_arch(arch) -> bytes:
    if len(arch.stages) > MAX_STAGES:
        raise ValueError(f"too many stages ({len(arch.stages)} > {MAX_STAGES})")
    buf = bytearray()
    buf += _HDR_FIXED.pack(
        1,                       # hdr_version
        len(arch.stages),
        3,                       # stem_in_c
        arch.stem_kernel,
        arch.stem_stride,
        0,                       # pad
        arch.stem_out,
        arch.head_rank,
        arch.embedding_dim,
    )
    for s in arch.stages:
        splits = list(s.xca_splits) + [0] * (4 - len(s.xca_splits))
        if len(splits) > 4:
            raise ValueError(f"too many xca splits ({len(s.xca_splits)} > 4)")
        buf += _STAGE_REC.pack(
            s.channels, s.n_convnext, s.dw_kernel, s.rank, s.hidden,
            int(s.has_xca), int(s.xca_pos_embed), s.xca_pos_hw,
            len(s.xca_splits), s.xca_n_convs, s.xca_dw_kernel, s.xca_n_heads,
            0,
            splits[0], splits[1], splits[2], splits[3],
            int(bool(s.downsample_to)), s.downsample_kernel, s.downsample_stride, 0,
            s.downsample_to, 0,
        )
    # pad with empty stages up to MAX_STAGES so the header is fixed-size
    empty = _STAGE_REC.pack(*([0] * len(_STAGE_REC.format.replace("<", ""))))
    while len(buf) < ARCH_HDR_SIZE:
        buf += empty
    assert len(buf) == ARCH_HDR_SIZE, f"header size {len(buf)} != {ARCH_HDR_SIZE}"
    return bytes(buf)


def _unpack_arch(buf: bytes):
    from arch import StageCfg, FaceXArch
    fixed = _HDR_FIXED.unpack_from(buf, 0)
    (hdr_v, n_stages, stem_in_c, stem_kernel, stem_stride, _pad,
     stem_out, head_rank, embedding_dim) = fixed
    if hdr_v != 1:
        raise ValueError(f"unknown header version {hdr_v}")
    stages = []
    for i in range(n_stages):
        off = _HDR_FIXED.size + i * STAGE_REC_SIZE
        rec = _STAGE_REC.unpack_from(buf, off)
        (channels, n_cn, dw_k, rank, hidden,
         has_xca, has_pos, pos_hw,
         n_splits, n_convs, xca_dw_k, n_heads, _p1,
         s0, s1, s2, s3,
         has_ds, ds_k, ds_s, _p2,
         ds_to, _p3) = rec
        splits = (s0, s1, s2, s3)[:n_splits]
        stages.append(StageCfg(
            channels=channels, n_convnext=n_cn,
            dw_kernel=dw_k, rank=rank, hidden=hidden,
            has_xca=bool(has_xca), xca_splits=tuple(splits),
            xca_n_convs=n_convs, xca_dw_kernel=xca_dw_k,
            xca_n_heads=n_heads,
            xca_pos_embed=bool(has_pos), xca_pos_hw=pos_hw,
            downsample_to=ds_to if has_ds else 0,
            downsample_kernel=ds_k if has_ds else 2,
            downsample_stride=ds_s if has_ds else 2,
        ))
    return FaceXArch(
        name="loaded", stem_out=stem_out,
        stem_kernel=stem_kernel, stem_stride=stem_stride,
        stages=stages, head_rank=head_rank, embedding_dim=embedding_dim,
    )


def write_bin(path: str, arch, tensors_by_name: dict) -> int:
    """Pack arch + tensors into the v2 .bin file. Returns bytes written."""
    layout = tensor_layout(arch)
    arch_bin = _pack_arch(arch)
    arch_json = json.dumps(arch_to_dict(arch), separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"EFX2")
        f.write(struct.pack("<I", 2))
        f.write(arch_bin)
        f.write(struct.pack("<I", len(arch_json)))
        f.write(arch_json)
        f.write(struct.pack("<I", len(layout)))
        for name in layout:
            t = tensors_by_name[name]
            blob = t.detach().cpu().contiguous().numpy().astype("float32").tobytes()
            f.write(struct.pack("<I", len(blob)))
            f.write(blob)
    import os
    return os.path.getsize(path)


def read_bin_header(path: str):
    """Return (arch_obj, n_tensors, payload_offset, json_dict)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"EFX2":
            raise ValueError(f"not an EFX2 file: magic={magic!r}")
        (ver,) = struct.unpack("<I", f.read(4))
        if ver != 2:
            raise ValueError(f"unsupported version {ver}")
        arch_bin = f.read(ARCH_HDR_SIZE)
        arch_obj = _unpack_arch(arch_bin)
        (jn,) = struct.unpack("<I", f.read(4))
        arch_json = json.loads(f.read(jn).decode("utf-8"))
        (n,) = struct.unpack("<I", f.read(4))
        payload_off = f.tell()
    return arch_obj, n, payload_off, arch_json


def read_bin_tensors(path: str):
    """Yield (name, np.ndarray) — name from layout derived from binary header."""
    import numpy as np
    arch_obj, n, off, _ = read_bin_header(path)
    layout = tensor_layout(arch_obj)
    assert len(layout) == n, f"layout len {len(layout)} != n_tensors {n}"
    with open(path, "rb") as f:
        f.seek(off)
        for name in layout:
            (sz,) = struct.unpack("<I", f.read(4))
            buf = f.read(sz)
            arr = np.frombuffer(buf, dtype="<f4")
            yield name, arr
