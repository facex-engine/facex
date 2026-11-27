<p align="center"><img src="docs/logo.jpg" alt="FaceX" width="420"></p>

# FaceX

FaceX detects faces, aligns them using five landmarks and computes recognition embeddings locally. The default native pipeline pairs YuNet with EdgeFace XS. It is available from C, Go and Python. A separate browser SDK runs the detector and embedder in WebAssembly. The MobileFaceNet models, including Nano ONNX, have their own input and output contracts.

[Live demo](https://facex-engine.github.io/facex/demo/) · [Releases](https://github.com/facex-engine/facex/releases) · [Python SDK](python/README.md) · [Browser SDK](wasm/README.md) · [Go package](go/facex) · [Benchmarks and experiments](docs/research)

## Choose an integration

| Goal | Entry point | Data contract |
|---|---|---|
| Detect and recognize faces in an image | [C](include/facex.h) or [Python](python/README.md) | RGB image → boxes, five landmark pairs, optional 512-component EdgeFace embeddings |
| Handle concurrent recognition requests from Go | [EmbedPool](go/facex/pool.go) | Aligned RGB floats → independent results, with bounded queueing and context cancellation |
| Detect, align and verify in a browser | [WebAssembly SDK](wasm/README.md) | Canvas, image or video → faces, crops and embeddings. Serve both modules and both weights with the application |
| Run compact ONNX recognition | [MobileFaceNet Nano](MFN_README.md) | Aligned NCHW float32 → a 256-component unit vector |
| Search a gallery with several views per identity | [Go](go/facex/identity.go) or [Python](python/README.md#identify-a-person-across-multiple-enrollment-photos) | Maximum cosine across enrolled views per identity, with score and margin thresholds |
| Try mesh, landmarks, smile and anti-spoofing | [Hosted demo](https://facex-engine.github.io/facex/demo/) | The demo uses separate ONNX models and a browser runtime. [Demo sources](docs/demo) |

## Build the native runtime

From the repository root on x86-64 Linux, with Make and a C compiler:

```sh
bash download_weights.sh
make -j2 all libfacex.so
make test
```

The download script verifies the EdgeFace checkpoint before installing or reusing it. The default build targets the local CPU. For the portable FP32 reference executable, run `make cli-fp32 CFLAGS="-O3"`. CI builds this executable on Apple Silicon and compares concurrent Go worker replies with serial inference using the released EdgeFace weights. Enroll and score with the same model, preprocessing and runtime mode. Native mixed precision and FP32 can produce different scores.

To enable detection, export the March 2023 OpenCV Zoo YuNet model from this pinned revision:

```sh
python -m pip install numpy onnx
curl -fL https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -o data/yunet.onnx
python tools/export_yunet_weights.py data/yunet.onnx weights/yunet_fp32.bin
```

| File | Bytes | SHA-256 |
|---|---:|---|
| [EdgeFace XS, v1.0.0](https://github.com/facex-engine/facex/releases/download/v1.0.0/edgeface_xs_fp32.bin) | 7,133,584 | `d23f7b3cf8bcda5e451b48cf012e7dc6c4963d5703fafb8893f956d0203691d8` |
| YuNet source ONNX for the command above | 232,589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Exported YuNet binary | 212,940 | `bf2a36bd1a3be7310a1062a22726a089bfee79c29b3cac16e3ffe568d4433d69` |
| [MobileFaceNet Nano, facex-nano-1.0](https://github.com/facex-engine/facex/releases/download/facex-nano-1.0/facex_nano.onnx) | 802,377 | `bd0094143bb4c7d04a76dcdad4e332f5985847b921ec4d6678c3fd89aaea3889` |

## Run the detector and embedder

With both weights installed, try the included photo:

```sh
python -m pip install -e './python[image]'
python - <<'PY'
import numpy as np
from PIL import Image
from facex import FaceX

rgb = np.asarray(Image.open("data/test_face.jpg").convert("RGB"))
with FaceX(embed_weights="data/edgeface_xs_fp32.bin",
           detect_weights="weights/yunet_fp32.bin",
           input_policy="rectangle_bgr") as fx:
    for face in fx.detect(rgb):
        print(face["bbox"], face["score"], face["embedding"].shape)
PY
```

Pass `embeddings=False` for geometry only. Rectangle modes preserve the aspect ratio and return coordinates in the original image. The [Python integration guide](python/README.md) covers small faces, tracked video frames, gallery search and the REST API.

In C, call `facex_init(embed_path, detector_path, NULL)` for plain weights. An embedding-only process can pass `NULL` for the detector. `facex_embed` takes an aligned 112 × 112 RGB face in HWC order, scaled to `[-1,1]`, and writes **512 raw float32 components**. Use `facex_similarity` for cosine scoring. Separate handles can run concurrently. Join all calls before freeing a handle. [C API](include/facex.h)

In Go, supply the CLI and weights paths in `facex.Config`. `NewEmbedPool(config, workers, maxWaiting)` runs one persistent engine per worker. Size the worker count against the CPU and memory available to the application. [Go binding](go/facex/facex.go) · [Concurrent requests](go/facex/pool.go)

On a Linux CI runner, eight live native EdgeFace handles, each warmed with one embedding, increased anonymous RSS by a median of **76.6 MiB**, down from **117.9 MiB** at commit `5f10ce0` (three runs per revision). All 135 fixture-crop embeddings matched the previous revision byte for byte within each of the two build modes. These measurements cover model memory and numerical compatibility. [Comparison run](https://github.com/facex-engine/facex/actions/runs/46050152862)

### Binary CLI protocol

| Command | Each input | Each output |
|---|---|---|
| `./facex-cli data/edgeface_xs_fp32.bin --server` | 37,632 float32 values, interleaved RGB (HWC), range `[-1,1]` | 512 raw float32 values |
| `./facex-cli data/edgeface_xs_fp32.bin --batch` | Same dimensions and range, planar RGB (CHW) | 512 raw float32 values |
| `./facex-cli --detect --pipe --detect-weights weights/yunet_fp32.bin --no-embeddings` | One int32 width, int32 height and packed RGB bytes | int32 face count and that many fixed-size FaceXResult records |

The embedding commands process complete frames until EOF. Each little-endian request is 150,528 bytes and each reply is 2,048 bytes. Keep diagnostics on stderr.

## Models and measurements

The released **facex_nano.onnx** has input **input [1,3,112,112]** and output **embedding [1,256]**. Use five-point ArcFace alignment and planar RGB float32. The training and evaluation scripts convert each 8-bit channel with `(pixel - 127.5) / 128.0`. The graph applies L2 normalization. The original release notes listed 512 dimensions. The hash above identifies the 256-dimensional artifact. [Tagged source](https://github.com/facex-engine/facex/tree/88465996e4cdd2e26d2169f5434bfa608a447b42) · [MobileFaceNet guide](MFN_README.md)

Historical project measurements on LFW used 10 folds. The threshold was selected on nine folds before evaluation on the held-out fold. For deployment, measure false accepts and false rejects on representative labeled pairs. [Evaluation script](training/scripts/lfw_eval.py)

| MobileFaceNet variant | Dimensions | Reported mean LFW accuracy | Fold standard deviation |
|---|---:|---:|---:|
| Nano | 256 | 95.62% | 1.11% |
| Tiny | 512 | 96.85% | 0.87% |
| Standard | 512 | 98.25% | 0.68% |
| XS | 512 | 99.07% | 0.40% |

The historical **3.0 ms** median on an Intel i5-11500 describes the XS recognition stage. Detection, alignment, gallery search and queueing add to end-to-end latency. The research directory records test conditions and workloads. [Runtime costs](docs/research/facex_runtime_costs.md) · [Concurrent requests](docs/research/multiface_requests.md) · [Detector budgets](docs/research/yunet_pixel_budget.md) · [Numerical comparisons](docs/research/math_25.md)

## Repository map

| Directory | Contents |
|---|---|
| [src](src), [include](include), [examples](examples) | Native libraries, public headers, CLI and C examples. The default EdgeFace engine reads EFXS, the v2 engine reads EFX2 and the MobileFaceNet engine reads EFM3 |
| [go/facex](go/facex), [python/facex](python/facex), [docker](docker) | Go workers and galleries, Python SDK, REST service |
| [wasm](wasm), [docs/demo](docs/demo) | Browser package, worker and hosted ONNX demo |
| [nn2](nn2/README.md) | Independent native object detector and NVR applications |
| [training](training/README.md) | Recognition training and exports, [landmarks](training/landmark), [3D landmarks](training/landmark3d) and [face detection](training/face_detect) |
| [tests](tests), [research](docs/research), [GitHub Actions](https://github.com/facex-engine/facex/actions) | Correctness checks, comparison scripts, benchmarks and CI results |

## Licensing and provenance

The top-level license is Apache 2.0. The detector header carries a CC BY-NC-SA 4.0 notice and NN2 has its own license file. Review the notices for the components and third-party assets included in a distribution. [Repository license](LICENSE) · [Detector header](include/detect.h) · [NN2 license](nn2/LICENSE) · [Upstream YuNet model](https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet)

The project identifies MS1M-RefineV2 in its recognition training recipe. The public Nano release materials do not identify an immutable dataset snapshot or a dataset permission record tied to the published SHA-256. For questions about commercial bundling, converted weights or redistribution, reference the artifact hash and intended use when requesting records. [Nano release](https://github.com/facex-engine/facex/releases/tag/facex-nano-1.0) · [Maintainer contact](mailto:bauratynov@gmail.com)
