# FaceX on i.MX NPU

A second build of FaceX (`libfacex_npu.so`) that dispatches inference
through the TensorFlow Lite C API. At runtime it picks the best available
delegate:

| SoC | Delegate | Library | NPU |
|---|---|---|---|
| **i.MX 8M Plus** | NXP VxDelegate | `libvx_delegate.so` | Verisilicon VIP9000, 2.3 TOPS |
| **i.MX 93** | Arm Ethos-U external delegate | `libethosu_delegate.so` | Arm Ethos-U65, ~0.5 TOPS |
| **i.MX 95** | NXP eIQ Neutron delegate | `libneutron_delegate.so` | NXP eIQ Neutron N3 |
| any AArch64 | XNNPACK (built-in) | (TFLite itself) | CPU only — slower |

Same C API (`facex_npu.h`), same `.tflite` artefacts (compiled offline),
same `libfacex_npu.so` source — the only thing that changes per board is
which delegate the runtime finds first.

> **Status:** the **embedder** path is fully wired. The **detector** path
> in `facex_npu_detect()` returns `-ENOSYS` today; the recommended
> deployment is the **hybrid pipeline** — CPU detector via `libfacex.a`
> + NPU embedder via `libfacex_npu.so`. See "Hybrid pipeline" below for
> the wiring.

---

## 1. Offline model conversion (one-time)

The NPU eats `.tflite`, not the FaceX `.bin` weights. Conversion runs once,
on a beefy host machine, and produces artefacts you ship to the board.

### Prereqs

```bash
pip install onnx2tf onnxruntime tensorflow numpy Pillow
pip install ethos-u-vela    # only for i.MX 93
```

Vela needs Python ≥ 3.10 and works best on Linux; it also runs on macOS
arm64 once the wheel is installed.

For **i.MX 95** the offline compiler is `neutron-converter` from NXP's
eIQ Toolkit (separate download, not on PyPI) — it consumes the same INT8
`.tflite` produced in step 2 and emits a Neutron-specialised `.tflite`.
Skip the Vela step entirely on i.MX 95 and run `neutron-converter` instead.

#### Obtaining neutron-converter

`neutron-converter` is shipped as part of NXP's **eIQ Toolkit** (the
machine-learning workflow add-on, not the BSP). It is *not* on PyPI,
GitHub, or any public package mirror — you have to download it from
nxp.com behind a free developer account.

1. Sign in (or register) at <https://nxp.com> with a developer account.
2. Go to **Software → Machine Learning → eIQ Toolkit**, or search the
   site for "eIQ Toolkit". Direct landing page:
   <https://www.nxp.com/design/design-center/software/eiq-ml-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT>
3. Download the installer that matches your host:
   - **Windows** — `.exe` installer (~2 GB), GUI + bundled CLIs.
   - **Linux** — `.deb` for Ubuntu 22.04 LTS or a generic tarball
     (un-tar to `/opt/nxp/eIQ_Toolkit_v<version>` and source the env).
   - macOS is **not officially supported** — use a Linux VM or run
     conversion on the same Linux host you build BSP images on.
4. Install. After install, open a shell and source the env script that
   ships in the install root (path varies by version):

   ```bash
   source /opt/nxp/eIQ_Toolkit_v1.*/bin/activate.sh
   neutron-converter --version    # sanity-check
   ```

   This puts `neutron-converter` (and a few helper tools — `eiq-cli`,
   model-zoo fetchers, profiler) on `PATH` and points its Python
   interpreter at the bundled venv. The shell session is the only thing
   that needs the env; once a `.tflite` is produced it's a plain file
   you can ship to the board.

5. With the env active, run the wrapper from this repo:

   ```bash
   tools/compile_neutron.sh weights/edgeface_xs_int8.tflite
   ```

   It calls `neutron-converter --target imx95 --output …_neutron.tflite`
   and writes alongside the input. The wrapper auto-detects the `_-`
   binary-name variants used across eIQ Toolkit versions; if your install
   uses a different flag layout the script's error message will tell you
   to consult `neutron-converter --help`.

> **Versioning note.** Pin the eIQ Toolkit release to the one whose
> `libneutron_delegate.so` matches what NXP's BSP ships on the target
> board. A converter from a newer toolkit can emit subgraph patterns
> the on-device delegate doesn't recognise, which manifests at runtime
> as "0 nodes delegated" even though the file *was* converted. The
> NXP BSP release notes list the matching toolkit version per release.

### Step 1: PyTorch → ONNX

The repo doesn't ship the EdgeFace PyTorch model — get it from the upstream
EdgeFace repo and export with the standard `torch.onnx.export(model, dummy,
"edgeface_xs.onnx", input_names=["input"], output_names=["embedding"],
opset_version=13)`. YuNet ships as ONNX in `weights/yunet_2023mar.onnx` —
no export needed.

### Step 2: ONNX → INT8 TFLite

```bash
python3 tools/onnx_to_tflite.py edgeface_xs.onnx weights/edgeface_xs_int8.tflite \
    --calib-dir calib_faces/

python3 tools/onnx_to_tflite.py weights/yunet_2023mar.onnx weights/yunet_int8.tflite \
    --calib-dir calib_faces/ --input-hw 160,160
```

`calib_faces/` should hold ~100 representative face crops (any size, any
format — they get resized + normalised inside the script). **Skipping
calibration is allowed but produces poor INT8 accuracy** — always provide
real images for production.

### Step 3 (i.MX 93 only): TFLite → Vela command stream

```bash
tools/compile_vela.sh weights/edgeface_xs_int8.tflite
tools/compile_vela.sh weights/yunet_int8.tflite
```

Outputs `weights/edgeface_xs_int8_vela.tflite` etc. — these are still
`.tflite` files but they contain the Ethos-U custom operator. Loading one
through TFLite + the Arm Ethos-U external delegate dispatches the heavy
ops to the NPU; anything Vela rejected is left on the CPU side of the
graph and runs in TFLite XNNPACK as usual.

The script prints op coverage from Vela's summary CSV — anything in the
"CPU" column is a layer that fell back. Common culprits: unsupported
activations (`GELU`, `swish`), dynamic shapes, ops that need FP32. Decompose
or replace, re-export, re-Vela.

i.MX 8M Plus skips this step — VxDelegate ingests the plain INT8 `.tflite`
directly. i.MX 95 uses `neutron-converter` from NXP's eIQ Toolkit instead
of Vela:

```bash
tools/compile_neutron.sh weights/edgeface_xs_int8.tflite
tools/compile_neutron.sh weights/yunet_int8.tflite
```

Outputs `weights/edgeface_xs_int8_neutron.tflite` etc. — these are the
files `libneutron_delegate.so` actually accelerates on the NPU. Loading
the un-converted INT8 file on a Neutron board still "works" (delegate
loads, model runs), but TFLite logs `0 nodes delegated` and inference
runs entirely on CPU/XNNPACK.

---

## 2. Building `libfacex_npu.so`

### On the host (dev / smoke tests)

Useful for syntax checks, the API smoke test, and running with the XNNPACK
fallback. Needs the TensorFlow Lite C library installed where your linker
can find it.

```bash
make imx-npu \
    TFLITE_INCLUDE=/opt/tflite/include \
    TFLITE_LIB=/opt/tflite/lib
```

If you don't have `libtensorflowlite_c.so` locally, build it from source
once (~30 min) following the Bazel instructions in
`tensorflow/lite/c/BUILD`, or grab the Python wheel that ships it
(`pip install tflite-runtime` extracts a usable `.so`, but C headers are
not included — you'll need to vendor `tensorflow/lite/c/c_api.h` from the
TF source tree).

### Cross-compiling for an i.MX board

Source the NXP Yocto SDK once (or pass it via `SDK=`):

```bash
make imx93   SDK=/opt/fsl-imx-xwayland/6.6-scarthgap
make imx95   SDK=/opt/fsl-imx-xwayland/6.6-scarthgap
make imx8mp  SDK=/opt/fsl-imx-xwayland/6.6-scarthgap
```

The Makefile sources the BSP's `environment-setup-aarch64-poky-linux`
script and hands `$CC` the right `-mcpu` flags for the target. NXP's
BSP already ships `libtensorflowlite_c.so` and the appropriate delegate
plugins under `/usr/lib/`, so the resulting `libfacex_npu.so` has
everything it needs at runtime on the device.

The three targets produce the same source artifact — the only differences
are the `-mcpu` tuning flags and which delegate the runtime ends up
choosing on each board.

---

## 3. API at a glance

```c
#include "facex_npu.h"

FaceXNpuOptions opts = { .verbose = 1, .num_threads = 4 };
FaceXNpu* fx = facex_npu_init("edgeface_xs_int8_vela.tflite",
                              NULL, /* detect — see hybrid pipeline */
                              &opts);
if (!fx) { /* check stderr — model missing, delegate failed, etc. */ }

printf("dispatch: %s\n", facex_npu_active_delegate(fx)); /* "neutron" / "ethos-u" / "vx" / "xnnpack" */

float emb[512];
facex_npu_embed(fx, aligned_face_112x112, emb);   /* float32 HWC, [-1,1] */

float sim = facex_npu_similarity(emb, reference_emb);
```

Full API in `include/facex_npu.h`. Mirrors `facex.h` so callers can
swap CPU and NPU backends at compile time.

---

## 4. Hybrid pipeline (recommended deployment)

The detector is small and CPU-cheap (~5 ms on A55 NEON via the existing
`libfacex.a`), the embedder is what benefits from NPU offload. Wire them
together at the application layer:

```c
#include "facex.h"        /* CPU detector */
#include "facex_npu.h"    /* NPU embedder */

/* Init both. CPU side without an embedder (passes NULL). */
FaceX*    cpu = facex_init(NULL, "weights/yunet_fp32.bin", NULL);
FaceXNpu* npu = facex_npu_init("weights/edgeface_xs_int8_vela.tflite",
                               NULL, NULL);

/* Per frame: detect on CPU, align on CPU, embed on NPU. */
DetectFace dets[10];
int n = facex_detect_only(cpu, rgb, w, h, dets, 10); /* TODO: helper to skip embed */
for (int i = 0; i < n; i++) {
    float aligned[112*112*3];
    align_face(rgb, w, h, dets[i].kps, aligned);    /* from libfacex */
    float emb[512];
    facex_npu_embed(npu, aligned, emb);
    /* compare emb against your gallery */
}
```

(`facex_detect_only` is a planned helper — for now use `facex_detect()`
and ignore the embedding it writes; it's a few microseconds wasted, not a
correctness issue.)

This is the layout the i.MX 93/95 sprint (B5 in the embedded port plan)
formalises. On i.MX 8M Plus the same wiring works — just a different
delegate gets selected at init time.

---

## 5. Testing

### Compile + link smoke (host)

```bash
make imx_npu_compile_test \
    TFLITE_INCLUDE=/opt/tflite/include \
    TFLITE_LIB=/opt/tflite/lib

./imx_npu_compile_test                            # API surface only, no model
./imx_npu_compile_test edgeface_xs_int8.tflite    # try a real init
```

The test is short on purpose: it validates `facex_npu_init` returns NULL
on bad input, that `facex_npu_active_delegate` reports a sensible value,
and that one `facex_npu_embed` call completes with finite output.

### Latency benchmark (`facex-bench-npu`)

`tools/bench_npu.c` is the TFLite-side companion to `facex-bench` — same
synthetic input pattern, same CSV schema, but inference runs through
the delegate stack. Useful for capturing CPU NEON vs XNNPACK vs Neutron
side-by-side without juggling two output formats.

```bash
make facex-bench-npu

# Auto-pick: tries Neutron → vx → Ethos-U → XNNPACK in order.
./facex-bench-npu --embed weights/edgeface_xs_int8.tflite

# Match NXP's `benchmark_model --external_delegate_path=…`:
./facex-bench-npu \
    --embed weights/edgeface_xs_int8_neutron.tflite \
    --external-delegate /usr/lib/libneutron_delegate.so
```

See `docs/benchmarking.md` for the full flag list and the "compare three
backends in one CSV" recipe.

### Hardware bring-up checklist

When you first plug in an EVK, the four sanity checks are the same shape
on every SoC — only the names change.

| Check | i.MX 93 | i.MX 95 | i.MX 8M Plus |
|---|---|---|---|
| Kernel config | `CONFIG_ARM_ETHOSU` | `CONFIG_NEUTRON` + `CONFIG_IMX_NEUTRON_REMOTEPROC` | `CONFIG_GALCORE` |
| `/sys/class/` entry | (driver-specific) | `/sys/class/neutron` | (driver-specific) |
| Device node | `/dev/ethosu0` | `/dev/neutron0` | `/dev/galcore` |
| Delegate `.so` | `libethosu_delegate.so` | `libneutron_delegate.so` | `libvx_delegate.so` |
| Firmware blob (if any) | — | `NeutronFirmware.elf` | (in-tree) |
| Offline compiler | `vela` | `neutron-converter` (eIQ Toolkit) | (none — VxDelegate ingests plain INT8) |
| Expected `active_delegate` | `ethos-u` | `neutron` | `vx` |

Then:

```bash
./imx_npu_compile_test embed_<board>.tflite
# prints e.g. "active delegate: neutron" on a healthy i.MX 95
```

If it prints `xnnpack` instead, the NPU delegate didn't `dlopen` —
re-run with `verbose=1` in `FaceXNpuOptions` and check `stderr`. Most
common causes: `.so` not on the loader path (fix with `ldconfig` or
`LD_LIBRARY_PATH`), kernel driver not loaded, or device node missing
permissions.

---

## 6. Known limitations

- **Detector path** — `facex_npu_detect` returns `-ENOSYS`. Use the hybrid
  pipeline above. Direct NPU detection requires a model-specific anchor
  decoder which we ship in `src/detect.c` (CPU side) but not in the NPU
  backend.
- **Hardware-untested** — the NPU code follows the published TFLite C API
  + delegate ABI. Compile is verified; runtime correctness on a real EVK
  is the next milestone — see the bring-up checklist in §5.
- **Vela op coverage** — `LayerNorm` and `GELU` aren't native Ethos-U65
  operators. Vela either decomposes them (slow but works) or kicks them
  to CPU. A model rewrite that uses `BatchNorm` + `ReLU6`-friendly
  activations would maximise NPU residency; until that lands, expect a
  few layers to run on the A55 cores.
- **Embedding sign convention** — the NPU backend always L2-normalises the
  output, regardless of whether the source `.tflite` ends with an L2 op.
  This makes cosine similarity behave identically to the CPU backend
  (`facex_similarity`).

---

## 7. See also

- `include/facex_backend.h` — the pluggable backend vtable. Long-term,
  CPU and NPU backends register through this; today they're separate APIs
  for clarity.
- `docs/implementation.md` — implementation details across all targets;
  §3 covers this i.MX library, §1 covers the bench tooling that exercises
  it on the host (XNNPACK fallback path).
- `docs/coverage_matrix.md` — current build/test status per SoC.
- `docs/mac.md` — Apple Silicon CPU build (NEON kernels). The same
  `libfacex.a` is what drives the CPU half of the hybrid pipeline above.
