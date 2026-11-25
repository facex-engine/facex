# Engine lifetime, working memory and optional recognition

Reference commit 9296653. Measurements are local Linux x86 experiments. The reported workspace totals, process RSS and API latency describe different resources.

## Engine lifetime

EdgeFace retained both sets of packed matrices and a cached position tensor. It also started three extra idle workers at every initialization, although the threaded GEMM branch is disabled. The loader now releases owned allocations on success and on partial initialization failure, and this engine no longer creates an unused pool. The standalone CLI and library use the same cleanup.

After warmup, process RSS was 10084 KiB after 10, 50 and 100 open/close cycles. A separate native test wraps allocation functions at link time: 100 successful cycles balanced every owned allocation. Injecting a failure at each of the 452 allocation sites reached in the measured initialization also left zero outstanding allocations. LeakSanitizer cannot enumerate threads in this sandbox under ptrace.

## YuNet workspace

The largest activation for each alternating buffer fits 16 channels at stride two or 64 at stride four. The objectness map needs a single channel at stride eight. The revised arena sizes follow those bounds. On a 640-square input, 1, 4 and 8 independent detector handles gave the following process-memory deltas after a forward pass on each:

| Handles | Virtual MiB before / after | Resident MiB before / after |
|---:|---:|---:|
| 1 | 82.3 / 19.9 | 19.9 / 19.9 |
| 4 | 329.8 / 79.9 | 79.6 / 79.5 |
| 8 | 659.1 / 159.3 | 158.8 / 158.8 |

Untouched virtual pages did not occupy physical memory in this environment. Full output comparisons against the reference cover 20 WIDER images at three shapes including odd sizes.

## REST without embeddings

Sequential Flask client posts with JPEGs containing 1, 8 and 32 repeated face fixtures; six timed requests per mode following warmup, alternating order. Times include request parsing, decoding and JSON serialization. The same engine and weights served both modes. Sizes below refer to representative JSON responses and include the timing field.

| Faces returned | Full median ms | Geometry median ms | Full bytes | Geometry bytes |
|---:|---:|---:|---:|---:|
| 1 | 50.44 | 45.70 | 11340 | 328 |
| 8 | 71.58 | 43.64 | 90681 | 2562 |
| 32 | 154.49 | 41.66 | 362085 | 9977 |

All returned boxes, scores and landmarks matched across modes. These are local service measurements, excluding external network and concurrent load.

## Guided refinement with the current small-face filter

BGR, minimum side eight. Reuse the 128 development and 320 evaluation images from the previous detail experiment for a paired retrospective comparison. Calibrate the guided score against the development baseline's unmatched count. The one-pass threshold is 0.75 and the chosen guided threshold is 0.78. The experiment admitted all suggested crops.

| Mode | Total matches | Small matches | Unmatched | Mean detector CPU ms |
|---|---:|---:|---:|---:|
| 640 single pass | 1022 | 563 | 33 | 49.38 |
| 640 plus conditional 448 crop | 1084 | 627 | 29 | 60.38 |

148 of 320 images triggered a crop. The net small-face gain was 64 (+11.4%), paired image bootstrap 95% interval +32 to +101. Unmatched delta −4, interval −13 to +5. Mean detector CPU rose 22%. These timings exclude recognition. The public SDK was compared to the reference refinement on a further 40 images, including 26 crop triggers; zero extra CPU budget reproduced the single-pass geometry.

```sh
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_guided_min8.py yunet.bin WIDER_val.zip wider_face_split.zip
make lib
gcc -Iinclude tests/weights_lifetime.c libfacex.a -Wl,--wrap=malloc,--wrap=calloc,--wrap=aligned_alloc,--wrap=free -lm -lpthread -o /tmp/weights_lifetime
/tmp/weights_lifetime edgeface_xs_fp32.bin
```
