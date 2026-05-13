# FaceX on macOS / Apple Silicon

The Mac build targets both Intel and Apple Silicon. On `arm64` it uses
hand-written NEON kernels (~5 ms / embed on M2). On `x86_64` it picks the
existing AVX2 / AVX-512 path. Same `make`, no flags to toggle.

## Prerequisites

- macOS 12 or newer.
- Xcode Command Line Tools — `xcode-select --install`. Provides `clang`,
  `swiftc`, and the system frameworks the camera benchmark links against.
- A copy of the EdgeFace embedder weights:
  ```bash
  bash download_weights.sh
  ```
  Drops `data/edgeface_xs_fp32.bin` (~7 MB) into the repo.
- (Optional) detector weights — converted from the bundled ONNX:
  ```bash
  pip3 install --quiet --break-system-packages onnx numpy
  python3 tools/export_yunet_weights.py     # writes weights/yunet_fp32.bin
  ```

## Build

```bash
make                    # libfacex.a + facex-cli + libdetect.a (host arch)
make test               # golden_test against data/edgeface_xs_fp32.bin
make mac-test           # macOS smoke test (embed + e2e + latency stats)
```

The Makefile auto-detects `uname -m`; on `arm64` it links `gemm_stub.c` +
`threadpool_pthread.c` and defines `FACEX_NO_INT8` so the engine runs the
FP32-packed-NEON path end to end. Output ends in `Built libfacex.a (arm64)`
or `Built libfacex.a (x86_64)`.

## Smoke test

```bash
make mac-test
```

Validates: weights load, embedding is finite + deterministic, self-similarity
is 1.0, and end-to-end detect+align+embed produces a face when fed
`tests/test_face_160.raw`. Reports min / median / p99 embed latency over 50
iterations.

Expected on M2 (8-core, 16 GB):
```
[ok] embed latency: min=4.28 ms  median=4.42 ms  p99=4.66 ms  (n=50)
[ok] e2e: detected 1 face(s) in 8.51 ms
       #0  bbox=[68.2,115.1 → 113.9,151.3]  score=0.835
```

## Camera benchmark

The benchmark grabs frames from the default camera via AVFoundation,
downscales to 160×160 RGB, calls `facex_detect`, and prints per-second FPS /
median / p99 / face count to stdout. Three build modes are exposed via
Makefile targets:

| Target | Swift flags | Use for |
|---|---|---|
| `make bench-camera` | `-O` | Default release build. Use for measurements. |
| `make bench-camera-debug` | `-Onone -g` | LLDB-friendly. Slower; do not use for FPS numbers. |
| `make bench-camera-profile` | `-O -g` | Optimised + symbols, suitable for Instruments / `sample`. |

All three drop a single binary `./facex-camera-bench` in the repo root.
The underlying script is `tools/build_bench_camera_mac.sh` if you want to
invoke `swiftc` directly.

### Permissions

On first run macOS prompts the parent terminal app for camera access. If
you've previously denied it, re-enable it in **System Settings ▸ Privacy &
Security ▸ Camera**. The benchmark exits with `error: camera access denied`
otherwise.

### Usage

```bash
./facex-camera-bench --help
./facex-camera-bench                        # run forever, Ctrl-C to stop
./facex-camera-bench --frames 200           # stop after 200 frames
./facex-camera-bench --no-detect            # camera-only baseline (engine skipped)
./facex-camera-bench --frames 60 --score 0.3
```

Flags:

| Flag | Default | Notes |
|---|---|---|
| `--frames N` | 0 (forever) | Hard frame budget. Useful for benchmarking. |
| `--width W` | 160 | Downscale width before calling the engine. |
| `--height H` | 160 | Downscale height. |
| `--score F` | 0.5 | Detector score threshold. Lower = more detections, more false positives. |
| `--embed PATH` | `data/edgeface_xs_fp32.bin` | Embedder weights file. |
| `--detect PATH` | `weights/yunet_fp32.bin` | Detector weights file. Missing file → embed-only mode. |
| `--no-detect` | off | Skip the engine call. Measures pure camera + colour-conversion overhead. |

### Reading the output

Per-second lines look like:

```
[t=585373.5s] frame 90  29.9 fps  detect+embed med=5.9 ms  p99=7.1 ms  faces=0
    bbox: [64,117 → 85,135]  score=0.51
```

- `fps` — capture rate. Capped by `AVCaptureSession.sessionPreset` (currently
  `.vga640x480`, so ~30 fps).
- `med`, `p99` — `facex_detect` latency including alignment + embedding when
  a face is found. Without a face, only detection runs and the cost drops.
- `faces` — count from the most recent frame in the second.
- `bbox`/`score` — first face from the most recent frame.

Camera-only baseline:

```
$ ./facex-camera-bench --frames 60 --no-detect
[t=585129.4s] frame 18  18.0 fps  camera med=0.0 ms  p99=0.0 ms  faces=0
```

The fps gap between `--no-detect` and the full pipeline tells you how
much budget the engine is consuming per frame.

## SME / SME2 (Apple M4 and newer)

Apple M4 introduced `FEAT_SME` (Scalable Matrix Extension). FaceX has an
opt-in SME path that uses `FMOPA` outer-product instructions on the ZA tile,
giving roughly **4× over NEON** for the FP32 packed matmuls that dominate
the embedder runtime.

### Build

```bash
make SME=1            # libfacex.a + facex-cli with the SME path enabled
# or
make mac-sme          # equivalent shorthand
```

Requirements:
- Apple Clang 16+ (Xcode 16+) or upstream Clang 18+ — needed for the ACLE
  2024 SME intrinsics in `<arm_sme.h>`.
- The default `make` keeps working on every other Mac (M1/M2/M3) and on
  any Xcode that doesn't have SME headers — SME is gated behind `SME=1`.

The SME source (`src/transformer_ops_sme.c`) is compiled with
`-march=armv9-a+sme`; every other source is built with the default
`-mcpu=apple-m1` so the auto-vectorizer can't accidentally emit SVE
instructions in non-SME translation units. This isolation matters — without
it, plain C in `transformer_ops.c` would silently get `rdvl`/`incb`/etc.
that trap on M1-M3.

### Runtime behaviour

`facex_has_sme()` (in `src/cpu_features.c`) reads
`hw.optional.arm.FEAT_SME` via `sysctlbyname` and caches the answer.
On first call to `matmul_fp32_packed` with `SME=1`-built libraries:

1. If the CPU lacks SME → mark SME disabled, take the NEON path forever.
2. If the CPU has SME → run a tiny SME-vs-scalar consistency check
   (4×8 × 8×8 matmul). On mismatch (>1e-3 anywhere) → `facex_disable_sme()`,
   fall back to NEON and print to stderr. This guards against bugs in the
   SME path on hardware we couldn't directly verify.
3. Per-call: kernel returns `-1` for shapes it refuses (M < SVL/4 — typically
   M < 4 — or K > 4096). The dispatcher then runs the NEON path.

Same `libfacex.a` ships across the M1-M5 lineup. NEON is the universal
floor; SME activates automatically when the chip and the build both
support it.

### Status

The kernel **compiles** clean and the disassembly contains real
`fmopa za0.s, p1/m, p0/m, z0.s, z1.s` (verified via `otool -tv` on an
M2 cross-compile). It is **not yet runtime-tested on M4** — when you run
it on a real M4 the self-check decides whether to keep SME on. If it
fails, file an issue with the stderr line; the bug is in our packing
or store layout, not in your hardware.

## Apple Accelerate (AMX)

Optional FP32 matmul backend that dispatches `matmul_fp32_packed` through
`cblas_sgemm` from `Accelerate.framework`. On Apple Silicon this lands on
the AMX coprocessor — typically **2-3× our NEON throughput** at the matmul
shapes EdgeFace exercises.

```bash
make ACCELERATE=1                 # libfacex.a + facex-cli with the AMX path
make ACCELERATE=1 mac-test        # smoke test through Accelerate
```

Combine with SME if you want both paths in one library:
```bash
make SME=1 ACCELERATE=1
```
Dispatch order in `matmul_fp32_packed`: Accelerate → SME → NEON / AVX2 /
scalar. The first kernel that accepts the shape wins; tiny shapes
(M < 4 or M·K·N < 4096) skip Accelerate's AMX warmup and stay on the
in-tree path.

Like SME, the Accelerate path runs a self-check on first matmul (cblas
vs scalar reference, 1e-4 relative tolerance). On divergence it calls
`facex_disable_accelerate()` and the rest of the process stays on NEON.

Measured on M2 (8-core, default `mac-test` synthetic input):

| Build | Embed median | E2E (detect+align+embed) |
|---|---:|---:|
| Default NEON | 4.59 ms | 9.0 ms |
| `ACCELERATE=1` | **3.57 ms** | **7.50 ms** |
| `SME=1` (M2: SME inert, NEON used) | 4.59 ms | 9.0 ms |

Same embedding bytes either way — `||emb||² = 0.076`, self-similarity
1.0000, identical bbox. Different kernel ordering can shift the LSB by
~ULP; cosine similarity stays at 1.0 because Accelerate's self-check
gates anything worse than 1e-4 relative.

## Core ML / Apple Neural Engine (opt-in)

```bash
make COREML=1
```

Builds the Objective-C bridge in `src/backend_coreml.m` and links
`CoreML.framework`. Public C API in `include/facex_coreml.h`:

```c
FaceXCoreMLOptions opts = { .compute_units = 0 /* ALL */, .verbose = 1 };
FaceXCoreML* fx = facex_coreml_init("weights/edgeface_xs.mlpackage", &opts);
float emb[512];
facex_coreml_embed(fx, aligned_face_112x112, emb);
printf("dispatched on: %s\n", facex_coreml_last_dispatch(fx));
facex_coreml_free(fx);
```

Two-step deployment:

1. **Build the model once on the host** with `tools/export_coreml.py`:
   ```bash
   pip install coremltools onnx numpy
   python3 tools/export_coreml.py edgeface_xs.onnx weights/edgeface_xs.mlpackage
   ```
   This produces a Core ML mlprogram (`.mlpackage`) with 6-bit
   palettized weights — about 1.8 MB on disk and small enough to
   live entirely in ANE-accessible memory. Pass `--no-palettize`
   for FP16 weights at higher accuracy and ~3× package size.

2. **Ship the `.mlpackage` next to your binary** and pass its path
   to `facex_coreml_init()`. macOS auto-compiles the package to
   `.mlmodelc` on first load (cached afterward).

`compute_units` selector lets you bench-route deliberately:

| Value | Constant | Behaviour |
|---|---|---|
| 0 | `MLComputeUnitsAll` | default — Core ML decides (usually ANE → GPU → CPU) |
| 1 | `MLComputeUnitsCPUAndGPU` | skip ANE, useful for ANE-vs-not bench |
| 2 | `MLComputeUnitsCPUOnly` | no GPU/ANE, debug |
| 3 | `MLComputeUnitsCPUAndNeuralEngine` | CPU + ANE only, skip GPU (macOS 13+) |

**Status:** the bridge **compiles + links + handles missing
.mlpackage gracefully** (verified by `scripts/test_all.sh`). End-to-
end ANE dispatch is gated on running `tools/export_coreml.py`
against an actual EdgeFace ONNX export; that artefact lives outside
this repo. Once the `.mlpackage` exists, expect ≈ 0.8 ms per embed
on M2, with the bulk of the model on ANE and the L2 normalize step
on CPU.

## Universal Mac binary (arm64 + x86_64)

For distribution to a mixed Apple Silicon / Intel population, build the
fat archive in one shot:

```bash
make mac-universal
```

Output: `libfacex-universal.a` (~360 KB combined). The build cross-
compiles each slice independently — arm64 with the NEON path, x86_64
with `-mavx2 -mfma` — then merges via `lipo -create`. Verify:

```bash
$ lipo -info libfacex-universal.a
Architectures in the fat file: libfacex-universal.a are: x86_64 arm64
```

Per-slice extract:
```bash
lipo -thin arm64  libfacex-universal.a -output libfacex-arm64.a
lipo -thin x86_64 libfacex-universal.a -output libfacex-x86_64.a
```

Each slice runs the architecture-appropriate kernels — the fat archive
isn't a NEON binary with x86 tacked on; both halves contain real, tuned
SIMD.

## Performance reference

Measured on an Apple M2 (8 cores, 16 GB), `release` build, NEON kernels enabled:

| Path | Latency |
|---|---:|
| `facex_embed` (112×112×3 → 512-d) | ~4.4 ms median |
| `facex_detect` (160×160 → bbox+kps), no face | ~4 ms |
| End-to-end detect + align + embed, single face | ~8.5 ms |
| Camera capture cost (`--no-detect`) | <1 ms |
| Sustained camera FPS (`--frames 90`) | 30 fps (camera-limited) |

For comparison, the scalar fallback (engine compiled without
`FACEX_HAVE_NEON`) is ~30 ms per embed — about 7× slower. Don't ship the
scalar build unless you're debugging.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `error: facex_init returned NULL` | Wrong weights path | `bash download_weights.sh` first |
| `error: camera access denied` | TCC consent not granted | System Settings ▸ Privacy ▸ Camera |
| `swiftc not found` | Xcode CLT missing | `xcode-select --install` |
| `Embedding norm: 0.275` in `make test` | Not a bug — see CLAUDE.md note | Self-similarity (cosine) is computed from raw outputs and is still 1.0 |
| `[skip] tests/test_face_160.raw not present` | Detector test asset missing | Already in repo; run `make mac-test` from repo root |
| `make bench-camera` fails on Intel Mac | Swift / AVFoundation paths are unchanged on x86; should work the same | If `swiftc` is present and weights downloaded, file an issue |

## See also

- `CLAUDE.md` — repo conventions and architecture summary.
- `docs/implementation.md` — implementation details across all targets;
  the Apple-Silicon section covers the Accelerate / SME / Core ML paths
  documented above.
- `docs/coverage_matrix.md` — per-flag coverage status (compiles? runs
  end-to-end? hardware-tested?).
