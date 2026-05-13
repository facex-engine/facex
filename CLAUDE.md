# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FaceX is a face detection + recognition library written in pure C99 with handwritten SIMD kernels (AVX2 / AVX-512 / VNNI on x86, NEON on AArch64). The default `libfacex.a` build has zero external dependencies — links only `libc` / `libm` / `libpthread`. It targets four deployment surfaces from a single C codebase:

- **Native** — `libfacex.a` static library + `facex.h` (Linux/macOS/Windows, x86-64 with AVX2 or AArch64 with NEON).
- **Browser** — Emscripten-built WASM (`facex.wasm` ≈ 74 KB, `detect.wasm` ≈ 28 KB) consumed via `wasm/facex-sdk.js`.
- **Go** — `go/facex` driving `facex-cli` over a stdin/stdout subprocess protocol.
- **Python** — `python/facex` with a ctypes path (preferred) and a subprocess fallback through `facex-cli`.

Model: EdgeFace-XS (1.77M params, 512-dim L2-normalized embedding, 99.73% LFW). Detector: YuNet today (`weights/yunet_*`).

## Build & test

Native build is driven by `Makefile`:

```bash
make                  # libfacex.a + facex-cli + libdetect.a
make example          # builds facex-example (links libfacex.a)
make encrypt          # builds facex-encrypt
make test             # builds + runs golden test against data/edgeface_xs_fp32.bin
make mac-test         # macOS smoke test (embed + e2e + latency)
make bench            # cross-platform synthetic latency bench (md/csv/json)
make bench-camera     # macOS camera benchmark (Swift + AVFoundation)
make ACCELERATE=1     # opt-in: dispatch matmul_fp32_packed through cblas_sgemm (AMX)
make SME=1            # opt-in: M4+ SME (FMOPA-based outer-product matmul)
make COREML=1         # opt-in: Core ML / ANE bridge (loads .mlpackage)
make mac-universal    # fat arm64 + x86_64 libfacex-universal.a for distribution
make imx-npu          # build libfacex_npu.{so,dylib} (TFLite + NPU delegate)
make imx93   SDK=...  # cross-compile NPU build for i.MX 93 (Ethos-U65)
make imx95   SDK=...  # cross-compile NPU build for i.MX 95 (eIQ Neutron N3)
make imx8mp  SDK=...  # cross-compile NPU build for i.MX 8M Plus (VxDelegate)
make clean
scripts/test_all.sh   # run every test runnable on this host
scripts/bench_all.sh  # sweep build flavours, produce one Markdown comparison table
```

The Makefile auto-detects AVX-512/VNNI via `gcc -mavx512f -dM -E - </dev/null` and adds `-mavx512f -mavx512vnni -mprefer-vector-width=512` when present. On Windows it links `-lsynchronization` and adds `.exe`. On `arm64` (Apple Silicon, AArch64 Linux) it switches to NEON kernels, links `gemm_stub.c` + `threadpool_pthread.c`, and defines `FACEX_NO_INT8` so the engine takes the FP32-packed path.

Weights are not in git (`.gitignore` excludes `*.bin`). Bootstrap with:

```bash
./download_weights.sh    # fetches data/edgeface_xs_fp32.bin (7 MB)
```

Tests live in `tests/` and are compiled ad-hoc against `libfacex.a`. There is no test runner / CI config in upstream — `scripts/test_all.sh` is the single-host harness.

## Architecture

### Unified C API (`include/facex.h`, `src/facex.c`)

`FaceX*` is the single user-facing handle. `facex_init(embed_weights, detect_weights, license_key)` initializes both engines (detector is optional — pass `NULL` for embed-only). `facex_detect` runs the full pipeline:

```
detect_run() → for each face: align_face() → engine_forward() → L2-normalize
```

Note: `src/facex.c` `#include`s `edgeface_engine.c` directly (single-translation-unit pattern) under `-DFACEX_LIB`. The CLI target compiles `edgeface_engine.c` separately. If you split or move embedder symbols, both build paths must stay consistent.

### Compute layers

- `src/transformer_ops.c` — LayerNorm, GELU (A&S 7.1.26 polynomial erf), packed FP32 MatMul (NR=8 AVX2 / NR=16 AVX-512 / hand-written NEON for AArch64), Conv kernels.
- `src/gemm_int8_4x8c8.c` — INT8 GEMM microkernel using `vpmaddubsw` (AVX2) / `vpdpbusd` (AVX-512 VNNI). x86-only; on ARM the `src/gemm_stub.c` shim is linked instead and the engine takes the FP32-packed path.
- `src/threadpool.c` — work-stealing pool using `WaitOnAddress` (Windows) / futex (Linux). `threadpool_pthread.c` is the macOS / generic-POSIX implementation; `threadpool_stub.c` and `threadpool_wasm.c` cover the rest.
- `src/edgeface_engine.c` — forward pass: stem → 4 stages of ConvNeXt blocks (stages 1-3 add XCA attention) → GAP → LN → FC → L2.
- `src/weight_crypto.c` — AES-256-CTR weight encryption with hardware binding. Doubles as a CLI when built with `-DWEIGHT_CRYPTO_MAIN`.
- `src/detect.c` — YuNet-style detector.
- `src/align.c` — 5-point affine warp from detector keypoints to canonical 112×112 (ArcFace template).

### Cross-platform shims

`include/compat.h` and `include/wasm_compat.h` paper over Windows/Linux/macOS/WASM differences (synchronization primitives, intrinsics availability, `aligned_alloc`). Prefer adding to these headers over `#ifdef` chains in source files.

### Benchmark tooling (`tools/bench.c`, `scripts/bench_all.sh`)

`tools/bench.c` is the cross-platform synthetic latency bench — same source compiles on macOS arm64/x86, Linux aarch64, future i.MX targets. Three output formats (`md`, `csv`, `json`) emitting the same data. Reports compiled-in vs runtime-active backends so the same binary tells you what's actually dispatching.

`scripts/bench_all.sh` sweeps build-flag combinations, runs `facex-bench` against each, and produces a unified Markdown comparison table. `tools/bench_camera_mac.swift` (built via `make bench-camera`) is the live-camera companion: AVFoundation capture → engine → per-second FPS / median / p99 / face count, with `--summary` mode emitting one CSV row in the same schema as the synthetic bench so the two can be merged.

See `docs/benchmarking.md` for the full guide.

## Conventions specific to this repo

- **No headers for internal C files.** `src/facex.c` `#include`s `edgeface_engine.c`. Keep static helpers static; add to `include/*.h` only when crossing the public ABI.
- **All public C identifiers are `facex_*` or `detect_*`.** Don't add new top-level prefixes.
- **Embedding contract:** input is 112×112×3 float32 HWC in `[-1, 1]`; output is 512 float32 L2-normalized. `facex_similarity` assumes both inputs are already normalized — no defensive renormalization.
- **Threshold defaults:** detector score 0.5, NMS IoU 0.4, "same person" similarity > 0.3. These are the documented defaults; change them via `facex_set_*_threshold` rather than recompiling.
- **Weight files are gitignored** (`*.bin`, `*.enc`, `*.npz`). Don't commit them; reference them through `download_weights.sh` or env vars (`FACEX_EMBED_WEIGHTS`, `FACEX_DETECT_WEIGHTS`).
- **Detector is mid-rewrite.** Sprint plan in `docs/plan/detector_plan.md` is authoritative for direction. Files under `wasm/src/` are the ground-up rewrite; `wasm/detect_new.{js,wasm}` is the in-progress build artifact alongside the legacy `wasm/detect.{js,wasm}`.
- **Mac perf paths are opt-in flags, not the default.** `make ACCELERATE=1` adds AMX via `cblas_sgemm`, `make SME=1` adds the M4+ SME path, `make COREML=1` adds the Core ML / ANE bridge. Default `make` stays portable across M1-M5 and any Xcode version; the optional flags require Xcode 16+ for SME. See `docs/mac.md` for the full Mac story.
- **NPU build is a separate library.** `libfacex_npu.{so,dylib}` (built only when `FACEX_BACKEND_TFLITE` is defined) is a TFLite C-API wrapper that runtime-loads eIQ Neutron (i.MX 95) / VxDelegate (8M Plus) / Arm Ethos-U (i.MX 93) / XNNPACK. Source: `src/backend_tflite.c`, public API: `include/facex_npu.h`. Detector path is intentionally `-ENOSYS` today — see `docs/imx_npu.md` for the recommended hybrid pipeline (CPU detect via libfacex.a + NPU embed via libfacex_npu.so).
- **ESP32-P4 is an ESP-IDF component, not a Make target.** `components/facex/` is the IDF wrapper, `examples/esp32p4_camera/` is a runnable IDF project. Build via `idf.py build flash monitor`, not `make`. The MIPI-CSI capture path is real; the face-detection backend defaults to a Kconfig-selected `stub` (synthetic faces) because EdgeFace-XS doesn't fit P4 RAM in real-time. The production path needs EdgeFace-Nano + ESP-NN backend — see `docs/esp32p4.md`.

## Limitations to keep in mind

- AVX2 mandatory on x86; NEON mandatory on AArch64. No scalar-only build today.
- The bundled EdgeFace weights are CC BY-NC-SA 4.0; engine code is Apache 2.0. Don't bake non-commercial weights into commercial-licensed example artifacts.
- No face detection in `facex_embed`-only mode — callers must align to 112×112 themselves or use `facex_detect`.
