# Benchmarking FaceX

There are several bench tools in this repo, each measuring a different
thing. This page explains which tool does what, and how to produce a
**single unified comparison table** across multiple build flavours.

## What measures what

| Tool | Measures | Input | Output schema |
|---|---|---|---|
| `tools/bench.c` (`make bench` → `./facex-bench`) | CPU engine latency: `embed` and/or `detect+align+embed` (e2e) | Synthetic deterministic | md / csv / json (selectable) |
| `tools/bench_npu.c` (`make facex-bench-npu` → `./facex-bench-npu`) | TFLite-delegate latency (Neutron, Ethos-U, VxDelegate, XNNPACK) | Synthetic deterministic | md / csv / json (selectable) — same CSV schema as `facex-bench` |
| `tools/bench_camera_mac.swift` (`make bench-camera` → `./facex-camera-bench`) | Camera capture pipeline: AVFoundation → downscale → engine. End-to-end frame budget. | Live camera | per-second log + `--summary` CSV row |
| `tests/bench_detect.c` (`gcc … bench_detect.c -L. -lfacex`) | Detector latency only | Synthetic | text |
| `tests/test_mac.c` (`make mac-test`) | Smoke test with embedded latency stats | Synthetic | text + backend report |
| `wasm/bench.js` (`node wasm/bench.js`) | Embed latency under WASM | Synthetic | text |
| `bench_node.mjs` (`node bench_node.mjs`) | LFW **accuracy** (not latency) | LFW image pairs | text |

The first three share a CSV schema so they can be combined:

```
label,compiled,active,stage,iters,min_ms,median_ms,mean_ms,p95_ms,p99_ms,e2e_face
```

`tools/bench.c` is the right tool for "compare backends". `bench_camera_mac.swift`
is the right tool for "what's the actual frame-to-result latency in a real
camera app on macOS".

## The unified table — `scripts/bench_all.sh`

One command, sweeps the build flag combinations that apply to your host,
runs `facex-bench` against each, and prints either Markdown or CSV.

```bash
# Markdown (default), short run for sanity:
scripts/bench_all.sh --iters 50 --warmup 5

# Long run, CSV for spreadsheet ingest:
scripts/bench_all.sh --iters 500 --warmup 50 --format csv > bench.csv

# Pin the configs you care about:
scripts/bench_all.sh --configs "default,ACCELERATE=1"
```

Sample output on Apple M2 (4 build flavours × 2 stages = 8 rows):

```
# FaceX bench sweep

host: `Darwin / arm64`
iters: 50  warmup: 5  embed: `data/edgeface_xs_fp32.bin`  detect: `weights/yunet_fp32.bin`

| label | active | stage | min ms | median ms | mean ms | p95 ms | p99 ms |
|---|---|---|--:|--:|--:|--:|--:|
| default              | NEON                  | embed | 4.394 | 4.528 | 4.546 | 4.836 | 5.034 |
| default              | NEON                  | e2e   | 8.275 | 8.382 | 8.410 | 8.652 | 8.765 |
| ACCELERATE=1         | Accelerate(AMX)+NEON  | embed | 3.303 | 3.616 | 3.651 | 4.036 | 4.125 |
| ACCELERATE=1         | Accelerate(AMX)+NEON  | e2e   | 7.292 | 7.402 | 7.418 | 7.603 | 7.912 |
| SME=1                | NEON                  | embed | 4.412 | 4.599 | 4.626 | 4.812 | 5.509 |
| SME=1                | NEON                  | e2e   | 8.281 | 8.511 | 8.568 | 9.106 | 9.432 |
| SME=1+ACCELERATE=1   | Accelerate(AMX)+NEON  | embed | 3.425 | 3.535 | 3.597 | 3.936 | 4.003 |
| SME=1+ACCELERATE=1   | Accelerate(AMX)+NEON  | e2e   | 7.275 | 7.394 | 7.422 | 7.632 | 7.845 |
```

The "active" column shows which backends actually dispatched at runtime
— note `SME=1` on M2 shows `NEON` because `hw.optional.arm.FEAT_SME=0`
on this chip. On an M4 it would show `SME` (or whatever the dispatcher
selects after the self-check passes).

## `tools/bench.c` (engine-only synthetic bench)

Deterministic input, structured output. Same source / same schema across
every build flavour and OS — that's what makes the comparison table
honest.

```bash
make bench                                          # default flags
make ACCELERATE=1 bench                             # build with Accelerate path
make SME=1 bench                                    # build with SME path

./facex-bench --iters 200 --stage both --format md
./facex-bench --iters 1000 --stage embed --format csv > embed_only.csv
./facex-bench --iters 100 --format json | jq .
```

Flags:

| Flag | Default | Notes |
|---|---|---|
| `--iters N` | 100 | Measurement iterations |
| `--warmup K` | 10 | Untimed warmup runs |
| `--stage embed\|e2e\|both` | both | E2E requires the detector |
| `--format md\|csv\|json` | md | Always emits the same data, different shape |
| `--label STR` | `""` | Tag for the row (set by `bench_all.sh`) |
| `--embed PATH` | `data/edgeface_xs_fp32.bin` | Embedder weights |
| `--detect PATH` | `weights/yunet_fp32.bin` (use `''` to skip) | Detector weights |

## `tools/bench_npu.c` (TFLite-delegate latency)

Same synthetic input pattern as `facex-bench`, but inference runs through
`libfacex_npu.so` → TFLite C API → an external delegate. Lets a single
harness compare CPU NEON (`facex-bench`), XNNPACK (`facex-bench-npu`
fallback), and any installed NPU delegate (Neutron / Ethos-U / VxDelegate)
in one CSV.

```bash
# Build (depends on libfacex_npu.so — run `make imx-npu` first or cross-build).
make facex-bench-npu

# Plain INT8 model on whatever delegate is auto-picked:
./facex-bench-npu --embed weights/edgeface_xs_int8.tflite

# Pin XNNPACK to get a clean CPU/TFLite baseline:
./facex-bench-npu --embed weights/edgeface_xs_int8.tflite --delegate xnnpack \
    --label "tflite-xnnpack" --format csv

# Force the i.MX 95 Neutron delegate by absolute path (matches NXP's
# benchmark_model --external_delegate_path=… invocation):
./facex-bench-npu \
    --embed weights/edgeface_xs_int8_neutron.tflite \
    --external-delegate /usr/lib/libneutron_delegate.so \
    --label "neutron" --format csv
```

Flags:

| Flag | Default | Notes |
|---|---|---|
| `--embed PATH` | (required) | `.tflite` embedder model |
| `--iters N` | 100 | Measurement iterations |
| `--warmup K` | 10 | Untimed warmup runs |
| `--format md\|csv\|json` | md | Same output schema as `facex-bench` |
| `--label STR` | `""` | Tag for the row |
| `--delegate NAME` | (auto) | `neutron` / `vx` / `ethos-u` / `xnnpack` / `armnn` |
| `--external-delegate PATH` | — | dlopen this `.so` directly, bypassing the registry. Wins over `--delegate`. |
| `--threads N` | autodetect | CPU threads for fallback layers |

The `compiled` column is fixed to `TFLite`; the `active` column is what
`facex_npu_active_delegate()` reports — `neutron`, `vx`, `ethos-u`,
`xnnpack`, or whatever name was derived from `--external-delegate`'s
basename.

E2E (`detect+align+embed`) is **not** in this bench: `facex_npu_detect`
is `-ENOSYS` today (see `docs/imx_npu.md` §6) and routing the detector
through the CPU `libfacex.a` here would conflate backends. For e2e
numbers on a hybrid CPU-detect / NPU-embed deployment, run
`facex-bench --stage e2e` (CPU only) and `facex-bench-npu --embed`
separately and add the medians.

### Comparing CPU NEON vs XNNPACK vs Neutron

```bash
make bench && make facex-bench-npu

./facex-bench --stage embed --format csv --label "neon" \
    | tee /tmp/cmp.csv

./facex-bench-npu --embed weights/edgeface_xs_int8.tflite \
    --delegate xnnpack --format csv --label "tflite-xnnpack" \
    | tail -n +2 >> /tmp/cmp.csv

./facex-bench-npu --embed weights/edgeface_xs_int8_neutron.tflite \
    --external-delegate /usr/lib/libneutron_delegate.so --format csv \
    --label "neutron" \
    | tail -n +2 >> /tmp/cmp.csv

# /tmp/cmp.csv now has three rows in one schema, ready for spreadsheet.
```

A common diagnostic outcome: the `neutron` row has latency identical to
the `tflite-xnnpack` row. That means the model wasn't compiled with
`neutron-converter` — TFLite logs `0 nodes delegated` and silently runs
on CPU. Fix: re-run the model through `tools/compile_neutron.sh` and
re-bench. See `docs/imx_npu.md` §1 for the full eIQ Toolkit setup.

## `tools/bench_camera_mac.swift` (live camera)

Different role: measures what happens with a real AVFoundation capture
pipeline, including the colour-format conversion and downscale. Pass
`--summary` to print one CSV row at exit using the same schema as
`facex-bench`, so `scripts/bench_all.sh` (or any spreadsheet tool) can
merge it with the engine-only numbers.

```bash
make bench-camera
./facex-camera-bench --frames 200 --summary --summary-label "camera-with-amx"
# → one CSV row to stdout, prefaced by per-second progress lines

# Just the camera capture, no engine work:
./facex-camera-bench --frames 200 --no-detect --summary --summary-label "camera-baseline"
```

Why a separate tool: AVFoundation, Core Image, and the macOS TCC camera
permission flow are macOS-only. The engine-only bench is portable.

## Combining engine and camera into one CSV

```bash
# Engine sweep:
scripts/bench_all.sh --iters 200 --format csv > /tmp/engine.csv

# Camera bench (twice — baseline and full pipeline):
./facex-camera-bench --frames 200 --no-detect --summary --summary-label "camera-base" 2>/dev/null \
    | tail -n +2 >> /tmp/engine.csv

./facex-camera-bench --frames 200 --summary --summary-label "camera-pipeline" 2>/dev/null \
    | tail -n +2 >> /tmp/engine.csv

# /tmp/engine.csv now has every row in one schema, ready for a spreadsheet.
```

The summary header is written to **stderr** by the camera bench precisely
so the second call's body can be appended without duplicating the header.
The engine bench writes its CSV header to stdout — which is why we use
`tail -n +2` on the camera output but not on the engine bench when
appending.

## When to choose which tool

| Question | Use this |
|---|---|
| "Which build flag gets me the lowest median embed latency on this Mac?" | `scripts/bench_all.sh` |
| "What's the p99 e2e latency under the Accelerate path?" | `make ACCELERATE=1 bench && ./facex-bench --iters 500` |
| "What's the actual frame-to-bbox time end-to-end with the camera?" | `./facex-camera-bench --frames 200 --summary` |
| "Does the new detector kernel still hit its budget?" | `tests/bench_detect.c` |
| "Did `facex_init` regress?" (cold start) | `make mac-test` (reports init + first-call latency) |
| "Did embedding accuracy regress?" | `bench_node.mjs` against LFW |

## Output schema reference

Every row emitted by `tools/bench.c --format csv` and by
`tools/bench_camera_mac.swift --summary`:

| Column | Type | Meaning |
|---|---|---|
| `label` | string | `--label` arg, or "camera" by default |
| `compiled` | string | `+`-joined list of FACEX_HAVE_* flags compiled in (`Accelerate+SME+NEON` etc.) |
| `active` | string | `+`-joined list of backends that actually dispatched at runtime |
| `stage` | enum | `embed`, `e2e`, or `camera` (no-detect baseline) |
| `iters` | int | Sample count for the percentiles |
| `min_ms` `median_ms` `mean_ms` `p95_ms` `p99_ms` | float | Latency stats |
| `e2e_face` | 0/1/`""` | 1 if at least one face was detected during the run, blank for embed-only |

## See also

- `docs/coverage_matrix.md` — what's compiled vs runtime-tested per arch
- `docs/mac.md` — Mac-specific build flags + perf reference
- `docs/imx_npu.md` — i.MX NPU bench notes (sprint A6/B6 hardware bring-up)
