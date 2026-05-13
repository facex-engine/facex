# FaceX MobileFaceNet — 4 size variants, Apache 2.0

Self-trained Apache-licensed face recognition weights for the FaceX
runtime. Four size points to fit different hardware budgets, all
sharing the same parametric C engine.

## Weights

| file | params | INT8 size | FP32 size | MS1M train acc |
|------|-------:|----------:|----------:|---------------:|
| `weights/facex_nano.bin` | 199K | ~200 KB | 827 KB | 15.4% |
| `weights/facex_tiny.bin` | 452K | ~450 KB | 1.86 MB | 25.1% |
| `weights/facex_standard.bin` | 968K | ~1 MB | 3.95 MB | 38.2% |
| `weights/facex_xs.bin` | 2.08M | ~2 MB | 8.42 MB | 50.9% |

All trained from scratch on MS1M-RefineV2 (5.82M images, 85,742 IDs)
with ArcFace (s=64, m=0.5), AdamW lr=1e-3 cosine, fp32. No upstream
pretrained weights — fully Apache-licensed.

## Architecture

Each is a MobileFaceNet (Chen et al. 2018) variant scaled by a width
multiplier. Topology:

```
Input 3x112x112
  -> Stem Conv 3x3 s=2 + BN + PReLU    (-> 56x56)
  -> DW Conv 3x3 s=1 + BN + PReLU
  -> Stage 1: 5x InvertedResidual t=2, s=2 first  (-> 28x28)
  -> Stage 2: 1x InvertedResidual t=4 s=2          (-> 14x14)
  -> Stage 3: 6x InvertedResidual t=2 s=1
  -> Stage 4: 1x InvertedResidual t=4 s=2          (-> 7x7)
  -> Stage 5: 2x InvertedResidual t=2 s=1
  -> Conv 1x1 + BN + PReLU                          (-> final_c)
  -> DW Conv 7x7 + BN (linear GDConv)               (-> 1x1)
  -> Conv 1x1 + BN                                  (-> emb_dim)
  -> L2 normalize
```

Width multipliers used: nano=0.36, tiny=0.55, standard=0.90, xs=1.35.
Embedding dim: 256 for nano, 512 for the rest.

## .bin format (EFM3)

Self-describing: a binary header (~80 bytes) names the stage shapes
and a JSON copy follows it for debugging. The engine reads only the
binary header.

```
"EFM3"          (4 bytes)
version u32 = 3
arch_header     (80 bytes — see binformat.py)
json_len u32 + JSON
n_tensors u32
[u32 size + FP32 bytes] x n_tensors
```

Tensor order is fixed by `binformat.tensor_layout(arch)` and is the
contract between `export_bin.py` and the C engine.

## C engine

`src/facex_mfn.c` — single-file parametric engine.

- Loads any of the 4 .bin files based on the embedded arch header.
- BatchNorm folded into the preceding conv at load time.
- AVX2 fast paths for 1x1 conv (the bulk of MFN compute), 3x3 DW,
  GDConv, PReLU, residual add. Plain-C fallback for stem.
- Single-threaded.

### Build

```bash
make mfn-cli          # standalone diagnostic CLI
make mfn-example      # tiny "embed + similarity" demo
```

### API

```c
#include "facex_mfn.h"

MfnEngine engine;
mfn_engine_init("weights/facex_standard.bin", &engine);

int D = mfn_embedding_dim(&engine);      // 256 (nano) or 512 (others)
float emb[512];
mfn_engine_forward(&engine, input_chw, emb);
// input_chw: [3 * 112 * 112] fp32, values in [-1, 1], CHW layout

float sim = mfn_similarity(emb_a, emb_b, D);
// > 0.3 typically = same person

mfn_engine_free(&engine);
```

### Image preprocessing

Same as InsightFace: 112×112 RGB, aligned (5-point), `(pixel - 127.5) / 128`,
CHW layout. You need an external detector (e.g. YuNet bundled in
`weights/yunet_*.onnx`) to align faces before feeding them in.

## Verifying a new checkpoint

```bash
cd training/scripts
python verify_bin.py --arch standard \
    --bin ../../weights/facex_standard.bin \
    --ckpt ../runs/standard/last.pt
```

Runs the .bin file through a numpy reference implementation of the
same op-graph as the C engine and compares to the PyTorch model. Max
expected error ~1e-5 (round-trip from fp32 file).

All four shipped models passed verification on commit:

```
nano:     max_err = 1.28e-05
tiny:     max_err = 1.64e-06
standard: max_err = 6.38e-06
xs:       max_err = 3.04e-06
```

## Training from scratch

See `training/README.md` and `training/RESUME.md` for the dataset prep
+ training pipeline. Realistic per-arch wall-time on a single RTX 5060
Ti (16 GB), fp32 training, ArcFace + MS1M:

| arch | epochs | per epoch | total |
|------|-------:|----------:|------:|
| nano | 40 | ~37 min | ~25 h |
| tiny | 35 | ~50 min | ~29 h |
| standard | 30 | ~80 min | ~40 h |
| xs | 30 | ~130 min | ~64 h |
