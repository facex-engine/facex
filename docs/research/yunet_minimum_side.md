# Does lowering the YuNet minimum box side help?

The earlier RGB ablation observed 17 additional small faces, with a paired 95% image bootstrap interval from −4 to +42. That result prompted this separate test of the current BGR detector.

Compare minimum predicted **width and height** of 16, 12 and 8 detector-canvas pixels, before NMS. The primary comparison is 8 versus 16 for the 640 BGR mode with recalibrated confidence. The 12-pixel cutoff and 800 mode are secondary. Count one-to-one matches against valid WIDER FACE validation boxes at IoU ≥0.50. Small faces have geometric mean side below 16 pixels after scaling the original long side to 320 pixels. This common annotation definition is distinct from the predicted-side cutoff.

For each size budget, compare (1) the same confidence threshold for every cutoff and (2) a threshold calibrated for each cutoff under the baseline count of unmatched predictions. Calibration maximizes small matches and then total matches. It uses 256 images from previously sampled sets. Evaluation uses all images outside the five named geometry, bottleneck, color and detail splits. Both modes use the same evaluation images. Calibration caps apply only on calibration data. Unmatched counts include detections on ignored annotations in this local protocol.

All modes keep BGR preprocessing, weights, NMS, landmark suppression and the 32-result cap fixed. One native forward pass supplies raw candidates per image and resolution. The replay is checked against separately compiled native implementations for each cutoff on every 83rd image. Paired percentile intervals use 4000 resamples of images. Intervals for secondary comparisons are descriptive and do not adjust for multiple comparisons.

## Reproduction

From a checkout containing ee217f5, with GCC, NumPy, Pillow, SciPy, YuNet 2023mar native weights and the two WIDER archives:

```sh
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_minimum.py yunet.bin WIDER_val.zip wider_face_split.zip --side 640
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_minimum.py yunet.bin WIDER_val.zip wider_face_split.zip --side 800
```

## Results

The 954 evaluation images contained 12730 valid faces, including 10935 small faces.

| Budget | Minimum side | Score | Total matches | Small matches | Unmatched |
|---|---:|---:|---:|---:|---:|
| 640 | 16 | 0.70 | 3437 | 1895 | 153 |
| 640 | 12 | 0.70 | 3965 | 2425 | 213 |
| 640 | 12 | 0.74 | 3848 | 2326 | 159 |
| 640 | 8 | 0.70 | 4316 | 2776 | 270 |
| 640 | 8 | 0.75 | 4064 | 2547 | 158 |
| 800 | 16 | 0.75 | 3932 | 2422 | 122 |
| 800 | 12 | 0.75 | 4354 | 2844 | 153 |
| 800 | 8 | 0.75 | 4516 | 3007 | 185 |

For the primary 640 comparison, lowering the minimum to 8 with calibrated score 0.75 gained 652 small matches (+34.4%, paired 95% interval +486 to +843). Small recall rose from 17.3% to 23.3%. Unmatched predictions increased by 5 (interval −20 to +33). At the same score 0.70 the size change alone gained 881 small matches but added 117 unmatched predictions. For minimum 12 with recalibration the gain was 431 (interval +317 to +567), with 6 additional unmatched predictions.

For budget 800 the original score 0.75 met the unmatched budget during calibration for both alternative minima. On evaluation, minimum 8 gained 585 small matches (interval +436 to +760) and added 63 unmatched detections (interval +40 to +89). This setting trades more false detections for recall. The earlier 17-face gain used RGB, different images and different thresholds. These results establish a benefit for the tested BGR pipeline and cannot isolate which of those changes explains the difference between studies.

The filter runs on detector predictions before suppression. It adds no network pass. Choose the minimum of 8 or 12 explicitly to enable it. The default remains 16. The native functions accept a minimum in detector pixels; the Python SDK and REST service expose the three evaluated choices. The 640 BGR policy automatically uses scores 0.75, 0.74 and 0.70 for sides 8, 12 and 16 respectively. All three use 0.75 for the 800 BGR policy. An explicit SDK score overrides the calibrated value. Recognition accuracy on the additionally detected tiny crops was not evaluated.
