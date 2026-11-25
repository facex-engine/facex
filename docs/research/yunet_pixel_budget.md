# YuNet input geometry at a fixed pixel budget

The baseline puts the entire RGB image in a 640 × 640 square with centered padding. We compared three alternatives without changing the detector weights or its native box filters.

| Mode | Input policy | Maximum pixels processed per image |
|---|---|---:|
| Square | One 640 × 640 pass | 409600 |
| Rectangle | One aspect preserving resize to a canvas aligned to 32, longest side at most 1280 | 409600 |
| Two tiles | Split the longer side in two and letterbox each part to 448 × 448 | 401408 |
| Four tiles | Split both sides in two and letterbox each part to 320 × 320 | 409600 |

Adjacent tiles overlap by approximately one eighth of the original side. Detections return to original image coordinates before a second, score ordered NMS at IoU 0.4 across tile outputs. A 1920 × 1080 image occupies 640 × 360 pixels in the baseline and 832 × 468 pixels on an 832 × 480 canvas in rectangle mode. This puts more of the same pixel budget into image detail.

## Protocol

WIDER FACE validation, 128 calibration images drawn with seed 725 from outside the fixed 640 image evaluation subset (seed 114). The evaluation subset was used in the earlier 25 task study. The four modes and their thresholds were compared in this experiment using disjoint calibration and evaluation images. Native detection runs at a common extraction threshold of 0.30. On calibration images the square uses a reporting threshold of 0.50. For each other mode choose the threshold with the most matches on a 0.01 grid subject to the square's unmatched detection count; ties use the lower threshold. Freeze the thresholds before evaluation.

One-to-one assignment at IoU ≥0.5 in original coordinates against valid boxes. An unmatched prediction counts as a false positive under this protocol, including predictions on ignored annotations. Small means a geometric mean of box width and height below 16 pixels after scaling the whole image's long side to 320, the same physical definition for all modes. Counts are operating points rather than official WIDER AP.

Linux x86, one pinned CPU thread, GCC with native optimization. CPU time includes crop, resize, canvas allocation, detection, coordinate conversion and cross-tile suppression. It excludes JPEG decoding, scoring and threshold selection. Mode order is randomized per image. Confidence intervals use 2000 paired bootstrap resamples of images. Timing at a common extraction threshold includes processing candidates later filtered by the calibrated threshold.

| Mode | Threshold | Calibration matches | Unmatched | Small matches |
|---|---:|---:|---:|---:|
| Square | 0.50 | 404 | 23 | 256 |
| Rectangle | 0.58 | 440 | 23 | 285 |
| Two tiles | 0.58 | 403 | 22 | 252 |
| Four tiles | 0.57 | 304 | 22 | 171 |

## Evaluation results

The 640 evaluation images contain 8909 valid faces, including 7902 small faces using the unrounded, aspect preserving reference scale. The earlier study rounded each resized side independently, which moves one borderline box between bins.

| Mode | Matches | Unmatched | Small matches | Mean CPU ms | Median CPU ms |
|---|---:|---:|---:|---:|---:|
| Square | 1570 | 200 | 796 | 74.504 | 73.320 |
| Rectangle | 1780 | 140 | 1011 | 74.405 | 72.902 |
| Two tiles | 1539 | 127 | 790 | 73.275 | 71.370 |
| Four tiles | 1271 | 119 | 563 | 74.129 | 72.644 |

Rectangle gained 210 matches and reduced unmatched detections by 60 with essentially identical mean CPU time. Small matches rose by 215 (+27.0% relative), from 10.1% to 12.8% recall. The other sizes accounted for five fewer matches. Paired 95% intervals for rectangle minus square were +146 to +284 total matches, −90 to −32 unmatched detections and +153 to +288 small matches. Direct native thresholding also reproduced post-filtered outputs across all four modes on 12 spot-check images.

The strongest tested operating point is a single rectangular pass at 0.58 for clients currently resizing to a fixed square. The two-tile mode gained precision but did not improve small-face recall at its calibrated threshold. Four tiles lost 233 small matches. These conclusions cover this model, image protocol and CPU. At the best operating point 6891 small faces still went undetected. The benchmark demonstrates an opt-in input policy; the native API continues to accept the caller's image dimensions.

## Reproduction

Use the native binary weights exported by tools/export_yunet_weights.py from YuNet 2023mar. Requires GCC, NumPy, SciPy and Pillow. Run from the repository root.

```sh
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_tiles.py yunet.bin WIDER_val.zip wider_face_split.zip
```

## Client integration

The Python SDK now exposes the measured policy through `FaceX(input_policy="rectangle")`.
The REST detection endpoint selects it by default. A run through the SDK on the
same 640 evaluation images with the REST limit of 32 faces per image produced
1759 matches, 131 unmatched detections and 990 small matches among 8909 valid
annotations. The experimental detector above allowed more results on crowded
frames. The SDK also clips coordinates to the original image bounds and computes
embeddings from the source image.
