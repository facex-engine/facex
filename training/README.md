# FaceX training

Train four MobileFaceNet recognition variants and export EFM3 weights. The training scripts describe MS1M-RefineV2 as their input. For the published Nano ONNX, see the [artifact contract](../MFN_README.md#public-nano-release) and [provenance notes](../README.md#licensing-and-provenance).

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
    binformat.py          - EFM3 reader/writer and tensor layout
    export_bin.py         - PyTorch checkpoint -> .bin
    smoke_test.py         - model -> .bin -> reload roundtrip check
```

Run the following commands from the `training/` directory. Training requires a separately obtained dataset.

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

The sequential trainer can resume interrupted runs from saved checkpoints. Report measured throughput with the dataset size, batch size, precision and GPU model. See the [run notes](RESUME.md) for the original environment.

## Export

After each training finishes (or manually for any checkpoint):

```bash
python scripts/export_bin.py --arch nano \
    --ckpt runs/nano/best.pt \
    --out ../weights/facex_nano.bin
```

The EFM3 file stores FP32 tensors. The C MobileFaceNet runtime can open it with `mfn_open()`. Use `mfn_embedding_dim()` to allocate the output for the chosen architecture. See the [native example](../examples/example_mfn.c).

## EFM3 format

The little-endian file starts with the four bytes `EFM3`, a uint32 version of 3, and an 80-byte architecture header. A length-prefixed UTF-8 JSON copy of the architecture and a uint32 tensor count follow. Each tensor record has a uint32 byte length and its contiguous FP32 data. The model exporter and loader must agree on tensor order. [Format implementation](scripts/binformat.py) · [Native loader](../src/facex_mfn.c)

## Evaluation

The [LFW evaluation script](scripts/lfw_eval.py) uses ten folds. The [project model table](../README.md#models-and-measurements) lists the historical scores and fold variation for the released variants. For experiments, keep the checkpoint, alignment, preprocessing and evaluation pairs with the result.
