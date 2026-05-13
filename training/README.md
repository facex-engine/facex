# FaceX training

Train 4 size variants (Nano / Tiny / Standard / XS) of the FaceX
embedding model from scratch on MS1M-RefineV2, export to the parametric
`.bin` v2 format.

## Files

```
training/
  data/
    ms1m/                 - converted MS1M (blob.jpg.bin + index.bin + meta.json)
    lfw-deepfunneled/     - LFW images (auto-downloaded)
    lfw.bin               - InsightFace-style verification bin (auto-built)
  runs/<arch>/
    last.pt, best.pt      - PyTorch checkpoints
    train.log             - per-iteration log
  scripts/
    arch.py               - 4 architecture configs + param counter
    model.py              - PyTorch model (mirrors C engine forward exactly)
    dataset.py            - mmap'd MS1M reader
    convert_tfrecord.py   - one-time: tfrecord -> blob+index
    prepare_lfw.py        - one-time: download LFW + build lfw.bin
    train.py              - ArcFace training for one arch
    train_all.py          - sequential trainer for all 4
    lfw_eval.py           - 10-fold CV verification
    binformat.py          - .bin v2 reader/writer + tensor layout
    export_bin.py         - PyTorch checkpoint -> .bin
    smoke_test.py         - model -> .bin -> reload roundtrip check
```

## One-time setup

```bash
# 1. Convert MS1M tfrecord (~30 min, ~16 GB blob)
python scripts/convert_tfrecord.py D:/archive.zip

# 2. Build LFW eval data (~5 min)
python scripts/prepare_lfw.py

# 3. Sanity check the pipeline before burning GPU
python scripts/smoke_test.py
```

## Training

```bash
# All 4, sequentially, with auto-resume on each
python scripts/train_all.py

# Or one at a time
python scripts/train.py --arch nano --epochs 25 --batch 512
```

Realistic wall time on RTX 5060 Ti (16 GB):

| arch     | params | epochs | est. time |
|----------|-------:|-------:|----------:|
| nano     |   203K |     25 |   ~4h     |
| tiny     |   514K |     25 |   ~6h     |
| standard |  1.07M |     22 |   ~10h    |
| xs       |  1.78M |     20 |   ~16h    |

Total ~36h. Train_all auto-resumes if interrupted.

## Export

After each training finishes (or manually for any checkpoint):

```bash
python scripts/export_bin.py --arch nano \
    --ckpt runs/nano/best.pt \
    --out ../weights/facex_nano.bin
```

The .bin is FP32 on disk (~4x the INT8 deploy size). The C engine's
INT8 quantization happens at load time.

## Targets

| arch     | LFW target |
|----------|-----------:|
| nano     | >= 97%     |
| tiny     | >= 98.5%   |
| standard | >= 99.5%   |
| xs       | >= 99.7%   |

## .bin v2 format

```
EFX2                     (4 bytes magic)
version (u32 = 2)
arch_json_len (u32)
arch_json (UTF-8 JSON describing stages/widths/kernels/ranks)
n_tensors (u32)
[tensor: u32 size + raw FP32 bytes] x n_tensors
```

The tensor order is fixed by `binformat.tensor_layout(arch)` and is the
contract between `export_bin.py` and the (parametric) C engine.

The legacy v1 `EFXS` format remains supported by the engine for the
existing EdgeFace-XS bundle; v2 is required for nano/tiny/standard
because the engine cannot infer their shapes from the file alone.
