"""
FaceX MobileFaceNet .bin v3 format — parametric, self-describing.

File layout (little-endian):
    magic        : 4 bytes  = b"EFM3"  (EdgeFaceX MFN v3)
    version      : u32      = 3
    binary header: BIN_HDR_SIZE bytes
    json_len     : u32      (debug copy of arch as JSON)
    json         : json_len bytes
    n_tensors    : u32
    tensor records: per-tensor [u32 size][raw FP32 bytes]

Binary header schema (fixed 32 bytes):
    u8  hdr_version (=1)
    u8  n_stages              (number of bottleneck stages)
    u8  stem_kernel
    u8  stem_stride
    u16 stem_channels
    u16 block_channels
    u16 final_channels
    u16 embedding_dim
    u8  gdc_kernel
    u8  pad[3]
    per stage (8 bytes each, MAX_STAGES=8):
        u8  expand
        u8  n_blocks
        u8  stride
        u8  pad
        u32 reserved
"""

import json
import struct

MAX_STAGES = 8
STAGE_REC_SIZE = 8
HDR_FIXED_SIZE = 16
ARCH_HDR_SIZE = HDR_FIXED_SIZE + MAX_STAGES * STAGE_REC_SIZE  # 80


_HDR_FIXED = struct.Struct("<BBBBHHHHB3s")    # 16 bytes
_STAGE_REC = struct.Struct("<BBBB4s")          # 8 bytes


# ---------- Tensor layout (Python <-> C contract) ---------------------------
#
# Each conv layer in MFN exports:
#   .conv.w     [Cout, Cin/groups, K, K]  FP32
#   .bn.w       [Cout]                    BN scale
#   .bn.b       [Cout]                    BN bias
#   .bn.mean    [Cout]                    BN running mean
#   .bn.var     [Cout]                    BN running var (used with eps to compute inv-std)
# Some conv layers add a PReLU:
#   .act.w      [Cout]                    PReLU per-channel slope
#
# Layer naming (in tensor_layout order):
#   stem.{conv.w, bn.{w,b,mean,var}, act.w}
#   dw_stem.{...}
#   stage{i}.block{j}.{expand, dw, project}.{conv.w, bn.{w,b,mean,var}}
#   stage{i}.block{j}.{expand, dw}.act.w        (project is linear, no act)
#   final_conv.{conv.w, bn.{w,b,mean,var}, act.w}
#   gdconv.{conv.w, bn.{w,b,mean,var}}          (no act)
#   embed_proj.{conv.w, bn.{w,b,mean,var}}      (no act)


def _conv_bn_names(prefix: str, with_act: bool):
    out = [f"{prefix}.conv.w",
           f"{prefix}.bn.w", f"{prefix}.bn.b",
           f"{prefix}.bn.mean", f"{prefix}.bn.var"]
    if with_act:
        out.append(f"{prefix}.act.w")
    return out


def tensor_layout(arch):
    names = []
    names += _conv_bn_names("stem", with_act=True)
    names += _conv_bn_names("dw_stem", with_act=True)
    c_in = arch.stem_channels
    for i, s in enumerate(arch.stages):
        c_out = arch.stem_channels if i == 0 else arch.block_channels
        for j in range(s.n):
            p = f"stage{i+1}.block{j}"
            names += _conv_bn_names(f"{p}.expand", with_act=True)
            names += _conv_bn_names(f"{p}.dw", with_act=True)
            names += _conv_bn_names(f"{p}.project", with_act=False)
        c_in = c_out
    names += _conv_bn_names("final_conv", with_act=True)
    names += _conv_bn_names("gdconv", with_act=False)
    names += _conv_bn_names("embed_proj", with_act=False)
    return names


def arch_to_dict(arch) -> dict:
    return {
        "name": arch.name,
        "width_mult": arch.width_mult,
        "stem_channels": arch.stem_channels,
        "block_channels": arch.block_channels,
        "final_channels": arch.final_channels,
        "embedding_dim": arch.embedding_dim,
        "gdc_kernel": arch.gdc_kernel,
        "stages": [{"t": s.t, "n": s.n, "s": s.s} for s in arch.stages],
    }


def _pack_arch(arch) -> bytes:
    if len(arch.stages) > MAX_STAGES:
        raise ValueError(f"too many stages")
    buf = bytearray()
    buf += _HDR_FIXED.pack(
        1, len(arch.stages), 3, 2,
        arch.stem_channels, arch.block_channels,
        arch.final_channels, arch.embedding_dim,
        arch.gdc_kernel, b"\x00\x00\x00",
    )
    for s in arch.stages:
        buf += _STAGE_REC.pack(s.t, s.n, s.s, 0, b"\x00\x00\x00\x00")
    while len(buf) < ARCH_HDR_SIZE:
        buf += _STAGE_REC.pack(0, 0, 0, 0, b"\x00\x00\x00\x00")
    assert len(buf) == ARCH_HDR_SIZE
    return bytes(buf)


def _unpack_arch(buf: bytes):
    from arch import StageCfg, FaceXArch
    fixed = _HDR_FIXED.unpack_from(buf, 0)
    (hdr_v, n_stages, stem_k, stem_s, stem_c, block_c, final_c, emb, gdc_k, _pad) = fixed
    if hdr_v != 1:
        raise ValueError(f"unknown header version {hdr_v}")
    stages = []
    for i in range(n_stages):
        off = HDR_FIXED_SIZE + i * STAGE_REC_SIZE
        t, n, sstride, _p, _r = _STAGE_REC.unpack_from(buf, off)
        stages.append(StageCfg(t=t, n=n, s=sstride))
    return FaceXArch(
        name="loaded", width_mult=1.0,
        stem_channels=stem_c, block_channels=block_c,
        final_channels=final_c, embedding_dim=emb,
        stages=stages, gdc_kernel=gdc_k,
    )


def write_bin(path: str, arch, tensors_by_name: dict) -> int:
    layout = tensor_layout(arch)
    arch_bin = _pack_arch(arch)
    arch_json = json.dumps(arch_to_dict(arch), separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"EFM3")
        f.write(struct.pack("<I", 3))
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
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"EFM3":
            raise ValueError(f"not an EFM3 file: magic={magic!r}")
        (ver,) = struct.unpack("<I", f.read(4))
        if ver != 3:
            raise ValueError(f"unsupported version {ver}")
        arch_bin = f.read(ARCH_HDR_SIZE)
        arch_obj = _unpack_arch(arch_bin)
        (jn,) = struct.unpack("<I", f.read(4))
        arch_json = json.loads(f.read(jn).decode("utf-8"))
        (n,) = struct.unpack("<I", f.read(4))
        payload_off = f.tell()
    return arch_obj, n, payload_off, arch_json


def read_bin_tensors(path: str):
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
