# Remaining detector, video and concurrency experiments

Reference: 6192e8e. All CPU results describe this Linux AMD EPYC 9V74 host.

## 4. Sparse detector geometry

YuNet calculates confidence first. Empty pyramid levels skip geometry. On levels with at most one accepted cell per 128 cells the detector computes the pointwise projection and subsequent 3×3 depthwise regression only around surviving centers. Regression of landmarks can also skip boxes rejected by geometry filters. Dense levels retain the dense path. The FMA implementation explicitly preserves dot-product accumulation order.

## 5. Stream the first depthwise convolution into pooling

Two scratch rows in the destination arena hold depthwise outputs. Pooling and ReLU consume the rows immediately. The next layer reads the pooled map without a pass over a full-size intermediate. Scalar borders and the vectorized row path are shared with the normal depthwise convolution.

On 193 WIDER validation images (one warmup, 96 measured per resolution) all output counts, coordinates, scores and landmarks matched bitwise across three builds. Work was interleaved on one pinned CPU thread, with 0.35 and 0.75 thresholds; GCC used x86-64-v3. Times cover native inference, not resizing, JPEG decoding or embedding.

| Budget | Original mean / median ms | Streamed pool mean / median | Pool and sparse heads mean / median |
|---|---:|---:|---:|
| 640 | 38.98 / 38.30 | 38.47 / 37.50 | 37.25 / 36.97 |
| 800 | 60.41 / 58.10 | 58.95 / 55.96 | 58.03 / 55.66 |

## 7. Refresh video embeddings on stable tracks

The optional Python cache consumes externally assigned track IDs. Its default refresh interval is at most 12 frames. Loss, a frame gap, overlapping or ambiguous tracks, poor detection confidence, small boxes and substantial changes in appearance or quality trigger a fresh embedding. A passing track ID alone does not establish identity.

We compared against fresh embeddings on the same matched detections from three annotated clips. For each 160-frame window we enrolled the first matched frame of each subject and classified the other 477 faces by nearest template. The David annotations start at source frame 300. Each window uses a fresh gallery and cache.

| Annotated frames per clip | Faces | Fresh embeddings with cache | Gallery disagreements | Wrong subject, fresh / cached |
|---|---:|---:|---:|---:|
| 1 through 160 | 480 | 102 | 0 / 477 | 1 / 1 |
| 161 through 320 | 480 | 126 | 13 / 477 | 13 / 14 |

Across these windows the cache avoided 732 of 960 embedding calls. All 13 changed decisions were on David. The test associates detections with ground truth and does not measure a production tracker's ID switch rate. A separate controlled crossing with two faces and a deliberate ID swap refreshed both tracks at the swap and produced zero gallery disagreements in 18 comparisons. Use fresh embeddings when every frame needs an independent recognition decision.

## 8. Concurrent inference

Use `preloaded:app` with Gunicorn preload and synchronous process workers. Packing is done in the parent; immutable weight pages can be shared using copy-on-write. The existing detector lock and native mutable scratch remain process-local. Thread-parallel use of a shared native handle was not enabled.

The harness executed 64 full-embedding Flask WSGI requests per configuration, in closed batches of 1, 2 or 4 concurrent requests, using eight distinct JPEG inputs. Each worker had one warmup request. Linux CPU affinity was restricted to the same four logical CPUs for all configurations. PSS sums the master and workers after warmup and accounts for shared physical pages. All six configurations produced the same per-image output hashes. The test uses pipes for request dispatch; external sockets, Gunicorn HTTP transport and network latency are outside the measurement.

| Workers | Lazy req/s | Preloaded req/s | Lazy / preloaded p95 ms | Lazy / preloaded PSS MiB |
|---:|---:|---:|---:|---:|
| 1 | 19.21 | 17.08 | 57.79 / 69.17 | 87.4 / 84.7 |
| 2 | 33.03 | 35.47 | 80.21 / 78.59 | 136.6 / 116.6 |
| 4 | 66.48 | 63.84 | 74.44 / 80.06 | 234.7 / 180.6 |

The primary benefit of preloading here is memory. Parallel processes account for the throughput gain. Each latency or throughput cell is a single local run.

## Reproduction

Set FACEX_LIB, EMBED_WEIGHTS, DETECT_WEIGHTS and the matching FACEX_EMBED_WEIGHTS and FACEX_DETECT_WEIGHTS before running. For the video test obtain the WebM and gt.txt files under david, dudek and faceocc2 from [OpenCV extra tracking data](https://github.com/opencv/opencv_extra/tree/5.x/testdata/cv/tracking). Place each clip and its annotation in its own named directory. Requires FFmpeg and ffprobe. The project does not distribute the clips.

```sh
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_heads.py yunet.bin WIDER_val.zip
PYTHONPATH=python python tests/bench_video_embeddings.py /path/to/sequences --start 0
PYTHONPATH=python python tests/bench_video_embeddings.py /path/to/sequences --start 160
PYTHONPATH=python:docker python tests/bench_prefork_detection.py 4 --preload
```
