# nn2 — Pure C YOLO Inference Engine

**1.50x faster than ONNX Runtime. Zero dependencies. 520KB binary. 70 cameras on one CPU.**

Built from scratch in pure C with hand-tuned AVX-512 SIMD. No frameworks, no libraries, no GPU required.

## Performance

### Single-frame inference (i5-11500, 6 cores / 12 threads)

| Input | nn2 | ONNX Runtime 1.23 | Speedup |
|-------|-----|-------------------|---------|
| 320x320 | **8.5ms (117 FPS)** | 12.7ms (78 FPS) | **1.50x** |
| 256x256 | **6.5ms (154 FPS)** | 8.4ms (119 FPS) | **1.30x** |
| 224x224 | **5.7ms (177 FPS)** | — | — |
| 192x192 | **4.8ms (208 FPS)** | — | — |

### Smart NVR pipeline (motion gate + Kalman tracking)

| Metric | Value |
|--------|-------|
| Average per frame | **0.56ms** |
| Effective throughput | **1,746 FPS** |
| Camera capacity (25fps) | **~70 cameras** |
| vs naive inference | **14.8x faster** |

Detection accuracy: 4/4 on bus.jpg (person 86%, bus 80%, person 72%, person 71%) — matches PyTorch reference.

## Quick Start

```bash
# Build everything
bash build.sh

# Single image detection
./nn2_img.exe weights/yolov8n_320.bin photo.jpg

# Production NVR server with web dashboard
./nn2_nvr_prod.exe -w weights/yolov8n_320.bin -p 8080
# Open http://localhost:8080 in browser
```

## Production NVR Server

Single binary with embedded web dashboard, REST API, smart pipeline:

```bash
./nn2_nvr_prod.exe -w weights/yolov8n_320.bin -p 8080
```

**Web Dashboard** at `http://localhost:8080` — live camera status, detection events, auto-refresh.

**REST API:**

```bash
# Health check
curl http://localhost:8080/api/health

# Add camera
curl -X POST http://localhost:8080/api/camera/add \
     -d '{"name":"Front Door","url":"rtsp://..."}'

# Detect objects in image
curl -X POST http://localhost:8080/api/detect --data-binary @photo.jpg

# Camera status + stats
curl http://localhost:8080/api/status

# Recent detection events
curl http://localhost:8080/api/events

# Simulate frame push (for testing)
curl -X POST http://localhost:8080/api/camera/sim
```

**Features:**
- Dynamic camera add/remove via API
- Motion-gated inference (skip static frames, ~0.05ms)
- Kalman tracking between detections (~0.01ms)
- SORT tracker with persistent object IDs
- Line-crossing people counting
- Zone intrusion detection (polygon, dwell time, class filter)
- JSON event logging
- Thread-safe shared model

## Export Weights

```bash
pip install ultralytics
python tools/export_yolov8n.py --size 320 --output weights/yolov8n_320.bin
python tools/export_yolov8n.py --size 256 --output weights/yolov8n_256.bin
```

## C API

```c
#include "nn2.h"

// Init (auto-detect threads)
NN2* ctx = nn2_init("weights/yolov8n_320.bin", 0);

// Detect objects
NN2Det dets[300];
int n = nn2_detect(ctx, rgb_hwc_uint8, width, height, dets, 300);

for (int i = 0; i < n; i++)
    printf("cls=%d score=%.0f%% box=(%.0f,%.0f,%.0f,%.0f)\n",
           dets[i].cls, dets[i].score * 100,
           dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);

nn2_free(ctx);
```

### NVR API

```c
#include "nn2_nvr.h"

void on_detect(const NN2NVREvent* e, void* ud) {
    printf("CAM %d: %d objects (%.1fms)\n",
           e->cam_id, e->det_count, e->inference_ms);
}

NN2NVR* nvr = nn2_nvr_create("weights/yolov8n_320.bin", 0);
nn2_nvr_add_camera(nvr, 0, "entrance");
nn2_nvr_set_callback(nvr, on_detect, NULL);
nn2_nvr_set_skip_frames(nvr, 5);

// Per frame:
nn2_nvr_push_frame(nvr, cam_id, frame_rgb, 640, 480);

nn2_nvr_free(nvr);
```

### Tracking + Counting + Zones

```c
#include "nn2_track.h"
#include "nn2_count.h"
#include "nn2_zone.h"

// Tracker: persistent IDs across frames
NN2Tracker* tr = nn2_tracker_create(30, 2);
NN2Track tracks[100];
int nt = nn2_tracker_update(tr, dets, n_dets, tracks, 100);

// Line counter: count crossings
NN2Counter* cnt = nn2_counter_create(0, 240, 640, 240);
nn2_counter_update(cnt, tracks, nt);
printf("In: %d, Out: %d\n", cnt->count_in, cnt->count_out);

// Zone intrusion: polygon + dwell time + class filter
float poly[] = {100,100, 400,100, 400,400, 100,400};
NN2Zone* z = nn2_zone_create("Restricted", 4, poly);
nn2_zone_set_classes(z, 1, (int[]){0}); // person only
nn2_zone_set_dwell(z, 3); // alert after 3 frames
NN2ZoneEvent events[10];
int ne = nn2_zone_check(z, tracks, nt, events, 10);
```

## Architecture

```
nn2/
├── include/
│   ├── nn2.h              Public detection API
│   ├── nn2_track.h         SORT tracker (Kalman + IoU)
│   ├── nn2_count.h         Line-crossing counter
│   ├── nn2_zone.h          Zone intrusion detection
│   ├── nn2_nvr.h           Multi-camera NVR pipeline
│   └── nn2_json.h          JSON event serialization
├── src/
│   ├── gemm.c              AVX-512 GEMM 6x32, KC-blocking, fused bias+SiLU
│   ├── gemm_res.c          Fused GEMM + residual (separate icache)
│   ├── conv.c              Conv dispatcher, parallel im2col
│   ├── conv_implicit.c     Zero-buffer 3x3 conv (no im2col)
│   ├── ops.c               AVX-512 maxpool, upsample, DW conv, concat
│   ├── decode.c            DFL decode + greedy NMS
│   ├── pool.c              Lock-free thread pool (WaitOnAddress)
│   ├── net.c               YOLOv8n forward pass (zero-copy C2f)
│   ├── nn2.c               Public API + preprocessing
│   ├── nvr_prod.c          Production NVR server + web dashboard
│   ├── track.c             SORT tracker implementation
│   ├── motion_gate.c       Motion detection (frame diff)
│   ├── smart_pipeline.c    Smart NVR pipeline (gate+track+skip)
│   ├── int8_nhwc/           INT8 VNNI pipeline (experimental)
│   └── specialized/         Compile-time specialized forward
├── tools/
│   ├── export_yolov8n.py   PyTorch → nn2 weight converter
│   ├── bench_compare.py    nn2 vs ONNX Runtime benchmark
│   └── prune_channels.py   Channel pruning tool
├── weights/
│   ├── yolov8n_320.bin     YOLOv8n @ 320 (12.6MB)
│   ├── yolov8n_256.bin     YOLOv8n @ 256 (12.6MB)
│   └── yolov8n_int8.bin    INT8 quantized (3.1MB)
└── build.sh                Build script
```

## Binaries

| Binary | Size | Purpose |
|--------|------|---------|
| `nn2.exe` | 260KB | CLI benchmark |
| `nn2_img.exe` | 488KB | Image detection (JPEG/PNG) |
| `nn2_nvr_prod.exe` | 520KB | Production NVR server |
| `nn2_nvr.exe` | 268KB | NVR simulation |
| `nanodet.exe` | 220KB | NanoDet-Plus-m |

## Key Optimizations

| Optimization | Effect |
|---|---|
| AVX-512 GEMM 6x32 with KC-blocking | Baseline, L1 cache fit |
| Fused GEMM + bias + SiLU | One memory pass (saves full C read+write) |
| A-packing (BLIS-style) | Sequential weight access, no stride |
| Implicit im2col | Eliminates 3.7MB buffer for large 3x3 convs |
| Parallel im2col | Multi-threaded im2col for 3x3 convs |
| AVX-512 maxpool + threading | SPPF 6x faster |
| AVX-512 upsample (permutexvar) | 16 pixels to 32 in one instruction |
| K-unroll x4 + prefetch | Better ILP and memory latency hiding |
| 2D tiling (M x N) | All cores busy even with few N-tiles |
| Fused residual (separate .c) | Saves memory pass without icache pollution |
| Cephes vectorized exp | Float-accurate SiLU (max error 4.8e-7) |
| Zero-copy C2f concat | Write bottleneck output directly to concat position |
| Lock-free thread pool | <500ns dispatch, spin-wait + WaitOnAddress |
| Motion gate | Skip static frames (~0.05ms instead of 8.5ms) |
| Kalman temporal skip | Track between detections (~0.01ms) |

## Requirements

- x86-64 CPU with AVX2+FMA (AVX-512 recommended)
- GCC 10+ or MinGW-w64
- ~64MB RAM for 320x320
- No other dependencies

## Optimization Journey

```
Naive C loop:              917.0 ms
+ AVX2 GEMM:              120.0 ms    (7.6x)
+ AVX-512 6x32:            44.0 ms    (2.7x)
+ Fused bias+SiLU:         26.0 ms    (1.7x)
+ Parallel im2col:         16.0 ms    (1.6x)
+ A-packing:               13.0 ms    (1.2x)
+ Implicit im2col:         12.6 ms    (1.03x)
+ AVX-512 maxpool:          8.9 ms    (1.4x)
+ KC-blocking + 2D tiling:  8.5 ms    (1.05x)
+ Fused residual:           8.1 ms    (1.05x)
─────────────────────────────────────────────
Total:         917ms → 8.1ms = 113x speedup
vs ONNX RT:    8.5ms vs 12.7ms = 1.50x faster
Smart pipeline: 0.56ms avg = 1,638x from start
```

## License

Apache 2.0. Free for commercial use, attribution required (see LICENSE).

Copyright (c) 2026 Baurzhan Atinov.

If you ship `nn2` in a product, an attribution line in your README or
About page is appreciated, not required by the license. Contributions
via PR welcome.
