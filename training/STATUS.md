# FaceX 4-model training — current status

## What's done in this session

### Python infrastructure (all in `training/scripts/`)
- `arch.py` — 4 architectures sized to budget. All 4 hit target ±8%:
  - nano: 203K params (target 200K)
  - tiny: 514K params (target 500K)
  - standard: 1.07M params (target 1M)
  - xs: 1.78M params (target 1.77M, matches existing EdgeFace-XS)
- `model.py` — PyTorch model mirroring C engine forward (Stem + 4 stages
  with ConvNeXt + optional XCA + LoRA-MLP + head). Same op order so
  exported weights load cleanly into the C engine.
- `dataset.py` — memory-mapped MS1M reader (no full-RAM load)
- `convert_tfrecord.py` — one-time tfrecord -> blob+index converter
- `prepare_lfw.py` — auto-downloads LFW + builds InsightFace-style lfw.bin
- `train.py` — ArcFace (s=64, m=0.5) + AdamW + cosine LR + bf16 + LFW eval each epoch
- `train_all.py` — sequential trainer for all 4 archs with auto-resume
- `lfw_eval.py` — 10-fold CV verification
- `binformat.py` — EFX2 .bin reader/writer with binary header
- `export_bin.py` — PyTorch checkpoint -> EFX2 .bin
- `smoke_test.py` — model -> .bin -> reload roundtrip check

### C engine (parametric)
- `src/edgeface_engine_v2.c` — new ~600-line parametric engine that loads
  EFX2 .bin files. Reads architecture from binary header in the .bin,
  dispatches forward pass dynamically. Reuses block primitives
  (convnext_block / xca_block / conv2d_hwc) from existing
  edgeface_engine.c (now exposed by removing `static`).
- `include/facex_v2.h` — public API for v2 (`v2_engine_init`,
  `v2_engine_forward`, `v2_engine_free`, `v2_embedding_dim`)
- `Makefile` — added `v2-cli` target → builds `facex-v2.exe`

### Format spec — EFX2 .bin v2

```
EFX2                               # 4 bytes
version (u32 = 2)
binary header (204 bytes)          # see binformat.py for schema
json_len (u32) + JSON header       # human-readable copy
n_tensors (u32)
[u32 size + raw FP32 bytes] x n_tensors
```

The binary header lets a single C engine instance handle any of the 4
architectures. Tensor order is fixed by `binformat.tensor_layout(arch)`.


## What's blocked

### Stuck: torch install (in progress, ~40% of 2.7 GB downloaded)
Network speed is ~5 MB/min — torch alone needs ~3 more hours to land.
After that:
- pip install completes
- smoke_test verifies model+export+reload roundtrip
- training can begin

### Pending: tensorflow-cpu install
Needed only by `convert_tfrecord.py` (once-off, ~30 min job).
Downloading in parallel with torch.

### Cannot fit in one session: actual GPU training
Realistic per-arch time on RTX 5060 Ti:
- nano:     ~4h
- tiny:     ~6h
- standard: ~10h
- xs:       ~16h
- **Total:  ~36h sequential**

## Next steps for the user

Once torch arrives (verify with `python -c "import torch; print(torch.cuda.is_available())"`):

```bash
cd D:/apps/facex/training/scripts

# 1. Verify pipeline (10 sec)
python smoke_test.py

# 2. Convert dataset (~30 min, one-off)
python convert_tfrecord.py D:/archive.zip

# 3. Build LFW eval data (~5 min)
python prepare_lfw.py

# 4. Train all 4 models sequentially (~36h, auto-resumes if interrupted)
python train_all.py

# 5. Verify weights with the v2 C engine (after `make v2-cli`)
../../facex-v2.exe ../../weights/facex_nano.bin
../../facex-v2.exe ../../weights/facex_tiny.bin
../../facex-v2.exe ../../weights/facex_standard.bin
../../facex-v2.exe ../../weights/facex_xs.bin
```

## Validation gates

After Nano finishes (~4h), check:
- LFW accuracy >= 96% (target 97%)
- `facex-v2.exe weights/facex_nano.bin` runs without errors
- Embedding values are reasonable (not all-zero / all-NaN)

If any of those fail, debug before proceeding to the larger models.

## Known untested areas

- C engine v2 has not been compiled (no GCC in this environment).
  User should run `make v2-cli` after torch install completes to verify
  the C side compiles cleanly.
- Smoke test has not been run (needs torch).
- LFW data builder uses simple center-crop+resize (not detector-based
  alignment); accuracy will be slightly lower than InsightFace's
  pre-aligned lfw.bin. For the cleanest eval, swap in the official
  InsightFace lfw.bin if available.
