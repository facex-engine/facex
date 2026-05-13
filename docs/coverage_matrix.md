# FaceX coverage matrix

What's compiled, what's syntax-checked, what's actually run end-to-end,
per (target × backend × build flag) combo. This document is filled from
`scripts/test_all.sh` output, not aspiration.

Each topic commit (Bench / Mac / i.MX / ESP32) appends its own rows.

## Legend

- ✅ **passing** — verified by `scripts/test_all.sh` on the noted host
- 🧪 **compile-only** — syntax-checked / linker-resolved against stub or vendor headers; not executed
- 🛠 **stub** — code path exists and compiles, but real backend or model is not implemented
- 🚫 **blocked** — needs hardware / SDK we don't have here
- — — does not apply

## Verification host

| Host | Hardware | Compiler |
|---|---|---|
| **mac-m2** | Apple M2, macOS 26 | Apple Clang 21 (Xcode 16+) |
| upstream | (assumed) | (varied) |

## CPU library (`libfacex.a`)

| Target / build flag | Compiles | Static analysis | Smoke test | E2E test | Tested on | Notes |
|---|---|---|---|---|---|---|
| `make` — host arch (Apple Silicon arm64, NEON) | ✅ | ✅ `otool -L` shows libSystem only | ✅ `golden_test`: `||emb||²=0.076`, sim=1.000 | ✅ via `facex-bench`: ~4.6 ms/embed median, ~8.4 ms e2e | mac-m2 | Default Mac build (Bench foundation) |
| `make` — host arch (x86-64 + AVX2) | 🧪 (upstream) | upstream | upstream `golden_test` | upstream | — | Pre-existing path, untouched |
| `make` — host arch (x86-64 + AVX-512 + VNNI) | 🧪 (upstream) | upstream | upstream | upstream | — | Auto-detected via `-mavx512f -dM` probe |
| `make` — Linux aarch64 (NEON) | 🧪 (Makefile path exists) | n/a | — | — | — | Same C as mac-m2; no Linux ARM box here |
| WASM (Emscripten) | 🧪 (upstream) | upstream | upstream demo | upstream demo | — | `wasm/` artifacts pre-existed |

## Apple Silicon — beyond NEON

| Target / build flag | Compiles | Static analysis | Smoke test | E2E test | Tested on | Notes |
|---|---|---|---|---|---|---|
| `make mac-test` (smoke + latency stats) | ✅ | ✅ | ✅ | ✅ — embed median ~4.6 ms, e2e ~9 ms, bbox `[68,115→114,151]` score 0.835 | mac-m2 | Reports compiled-in vs runtime-active backends |
| `make ACCELERATE=1` — Apple AMX via cblas_sgemm | ✅ | ✅ `nm` shows `matmul_fp32_packed_accelerate`; `otool -L` shows `Accelerate.framework` linked | ✅ self-check on init: cblas_sgemm vs scalar within 1e-4 relative | ✅ `mac-test`: 3.5 ms/embed (-22%), e2e 7.5 ms (-13%); embedding byte-equivalent | mac-m2 | Falls back to NEON for tiny shapes |
| `make SME=1` — Apple Silicon arm64 + SME (M4+) | ✅ | ✅ `fmopa` present in `transformer_ops_sme.o`; `rdvl/smstart/fmopa` absent from `transformer_ops.o` (M1-M3 safe) | ✅ `mac-test` byte-identical on M2 (SME inert because `facex_has_sme()=0`) | 🚫 needs M4 hardware | mac-m2 (SME path inert) | Self-check at first matmul disables SME on output divergence |
| `make SME=1 ACCELERATE=1` — both flags combined | ✅ | ✅ both libs link | ✅ | ✅ `mac-test` passes | mac-m2 | Accelerate wins per dispatch order; SME path inert on M2 |
| `make COREML=1` — Core ML / ANE bridge | ✅ | ✅ `nm` shows `facex_coreml_init/_embed/_free`; `otool -L` shows `CoreML.framework` linked | ✅ missing-`.mlpackage` smoke: returns NULL with clear stderr message, no crash | 🚫 needs real `.mlpackage` produced by tools/export_coreml.py from EdgeFace ONNX | mac-m2 (compile + link + error path only) | Obj-C bridge (`src/backend_coreml.m`); ARC-managed; supports `compute_units` hint |
| `make mac-universal` — fat arm64 + x86_64 binary | ✅ | ✅ `file` reports "universal binary"; both slices present | ✅ arm64 slice has 293 NEON insts (fmla/fmul); x86_64 slice has 786 AVX insts (vfmadd/vmovups) | n/a (smoke runs against thin host build) | mac-m2 | 358 KB combined; built via cross-compile + `lipo -create` |
| `tools/export_coreml.py` — ONNX → .mlpackage with INT8 palettization | ✅ | ✅ `--help` parses, AST validates | 🚫 needs ONNX EdgeFace export | — | mac-m2 (parses) | Calls `coremltools.convert(convert_to="mlprogram")` + `palettize_weights` for ANE INT8 |
| `tests/test_mac.c` (smoke + latency) | ✅ | ✅ | ✅ | ✅ — backend reporting works across all flag combos | mac-m2 | Prints "compiled in" + "active at runtime" lines |

## NPU library (`libfacex_npu.{so,dylib}`)

| Target / build flag | Compiles | Static analysis | Smoke test | E2E test | Tested on | Notes |
|---|---|---|---|---|---|---|
| `make imx-npu` — host TFLite + XNNPACK fallback | 🚫 (no libtensorflowlite_c on dev box) | ✅ `clang -fsyntax-only` against header stub | — | — | mac-m2 (syntax-only) | Real build needs TFLite C lib + headers |
| `make imx93 SDK=…` — A55 + Ethos-U65 (Vela) | 🚫 (no NXP SDK here) | 🧪 same syntax check | — | — | — | Compile-time path verified; runtime needs `/dev/ethosu0` |
| `make imx95 SDK=…` — A55 + Ethos-U65 | 🚫 (no NXP SDK here) | 🧪 same syntax check | — | — | — | Same artifact as imx93, different `-mtune` |
| `make imx8mp SDK=…` — A53 + VIP9000 (VxDelegate) | 🚫 (no NXP SDK here) | 🧪 same syntax check | — | — | — | Delegate selected at runtime via `dlopen` |
| `imx_npu_compile_test` — API smoke | ✅ syntax | ✅ | — | — | mac-m2 (syntax) | Runs once TFLite is on the host; checks NULL handling, dtype branches |
| **NPU embedder path** | — | ✅ | — | 🚫 needs board | — | Fully wired (INT8 quantize/dequantize + L2 norm) |
| **NPU detector path** | — | — | — | — | — | Returns `-ENOSYS` by design — use hybrid pipeline |
| `tools/onnx_to_tflite.py` (offline) | ✅ syntax | 🚫 (needs `onnx2tf` + `tensorflow`) | — | — | mac-m2 (parses) | Offline NPU model conversion |
| `tools/compile_vela.sh` (offline) | ✅ syntax | 🚫 (needs `ethos-u-vela`) | — | — | mac-m2 (parses) | i.MX 93/95 Vela compilation |

## ESP32-P4

| Target / build flag | Compiles | Static analysis | Smoke test | E2E test | Tested on | Notes |
|---|---|---|---|---|---|---|
| `components/facex/` ESP-IDF wrapper (host syntax) | ✅ stub backend syntax | ✅ `clang -fsyntax-only` against synthesized esp_err/log/timer/sdkconfig | — | — | mac-m2 (syntax) | Real build needs IDF v5.4+ |
| `examples/esp32p4_camera/` IDF project | 🚫 (no IDF here) | — | — | — | — | App-level, real ESP-IDF build target |
| FaceX backend: `stub` | ✅ | ✅ | — | — | mac-m2 (syntax) | Synthetic faces, default backend |
| FaceX backend: `native` | 🛠 | ✅ | — | — | — | EdgeFace-XS would link; impractically slow on P4 (1-3 s / frame) |
| FaceX backend: `espnn` | 🚫 reserved | — | — | — | — | Future — distilled EdgeFace-Nano + esp-nn kernels |

## Bench infrastructure

| Tool | Compiles / runs | Tested | Notes |
|---|---|---|---|
| `facex-bench` (cross-platform engine bench) | ✅ | ✅ md + csv + json output; embed and e2e stages | One source / one schema across all build flavours |
| `facex-camera-bench` (release, AVFoundation) | ✅ | ✅ — 29.0 fps, detect+embed med ~5 ms | Mac-only; lives in this commit but exercises the camera pipeline |
| `facex-camera-bench` (debug / profile) | ✅ | 🧪 builds; not benchmarked | LLDB / Instruments variants |
| `facex-camera-bench --summary` | ✅ | ✅ emits one CSV row at exit (schema matches `facex-bench`) | Lets camera and engine numbers join in one table |
| `scripts/bench_all.sh` (build-flag sweep) | ✅ | ✅ produces unified Markdown table comparing default config on M2 | Run before/after a perf change to spot regressions |
| `scripts/test_all.sh` (full local test runner) | ✅ | ✅ all checks runnable on mac-m2 pass | Topic commits amend with their own checks |

## Pre-existing tooling

| Tool | Compiles / runs | Tested | Notes |
|---|---|---|---|
| `bash download_weights.sh` | ✅ | ✅ produces `data/edgeface_xs_fp32.bin` | One-time fetch from GitHub release |
| `tools/export_yunet_weights.py` | ✅ (needs `onnx`+`numpy`) | ✅ produces `weights/yunet_fp32.bin` | Pre-existing |
| `facex-cli` | ✅ | ✅ via `make test` | stdin/stdout subprocess engine |
| `golden-test` | ✅ | ✅ — `||emb||²=0.076`, self-sim 1.000 | Cross-platform smoke |

## Test runner

`scripts/test_all.sh` runs everything in this matrix that's runnable on the
current host. Latest result on **mac-m2** at the Bench commit:

```
host:     Darwin / arm64
compiler: Apple clang version 21.0.0
ALL OK
```

The runner exits non-zero on first failure and prints the offending
command's output (first 20 lines), so CI can wire it directly.

## What's NOT covered (and why)

| Configuration | Why uncovered | Path to coverage |
|---|---|---|
| Apple-specific perf paths (Accelerate / SME / Core ML) | Added in Mac commit | See `docs/mac.md` once it lands |
| i.MX 93 / 95 / 8M Plus boards | NPU library lives in i.MX commit | See `docs/imx_npu.md` once it lands |
| ESP32-P4 dev kit | Component lives in ESP32 commit | See `docs/esp32p4.md` once it lands |
| Linux aarch64 server | No host available | Run `scripts/test_all.sh` on a Graviton / Ampere instance |
| Linux x86-64 with AVX-512+VNNI | No host with VNNI here | Same |

## Reading guide for maintainers

- "✅ tested on mac-m2" rows are real, current-tip. Reproduce with
  `scripts/test_all.sh`.
- "🧪 compile-only" rows are honest about *what* was checked and *what
  wasn't*. Don't promote to "✅" without an end-to-end run.
- "🚫 blocked" rows have a clear unblock path in the rightmost column.
- Topic commits (Mac, i.MX, ESP32) **append** new rows here — diff this
  file when shipping new platform support.
