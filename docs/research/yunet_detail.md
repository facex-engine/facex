# Recovering small faces with a larger detector input

The comparison uses YuNet 2023mar with BGR detector input, original RGB source coordinates and at most 32 results. A budget of 640 means that the aligned rectangular canvas has at most 640² pixels. The 800 budget admits up to 800² pixels (+56.25%). There is one whole-frame inference per image in either case.

## Operating points

- **Current:** 640 budget and threshold 0.70.
- **Soft:** same canvas and proposals, threshold 0.55. This isolates the cost in false positives of accepting weaker detections.
- **Full 800:** larger rectangular canvas with a threshold calibrated on development images.
- **Guided 448:** first run the 640 detector at score 0.35. Look for boxes smaller than 20 reference pixels with scores below 0.85, choose the half-frame neighborhood containing the most such hints, and center the second crop on its seed. Run it at a 448 pixel budget if there is a hint. Merge in source coordinates with IoU suppression at 0.4 and retain 32 results. This rule cannot revisit an area with no initial hint unless it lies inside the chosen crop.

The 128 development and 320 evaluation images were drawn with seed 60177 from WIDER FACE validation after excluding all samples used in the previous rectangular geometry and eight bottleneck experiments. The two calibrated variants select a threshold on the development set to maximize small matches without exceeding the current mode's unmatched count. Ties favor total matches and then closeness to 0.70. Thresholds stay fixed on evaluation images.

We count one-to-one matches at IoU ≥0.50 against valid boxes. Small means geometric mean side below 16 pixels when the original long side is scaled to 320 pixels. Recovered and lost counts compare the specific matched small annotation IDs per image. Percentile intervals resample entire images 2000 times. Unmatched counts use the same local protocol as earlier studies; these are not official WIDER AP results.

## Native CPU kernels

The 1×1 AVX2/FMA path holds four output channels and 16 positions in registers and reuses each loaded input across channels. Interior 3×3 depthwise convolution processes eight adjacent sites per vector. The stride-two stem gathers eight strided samples per tap. Accumulation follows the same channel and tap order; border and tail positions use the existing scalar code. Other CPU backends use the previous implementation.

CPU times include BGR resizing and detection on a single pinned Linux thread. JPEG decoding and embedding are outside the timer. The accuracy experiment uses the reference kernels and a common extraction threshold so all operating points can be recalibrated from the same detections. The separate paired speed comparison measures the previous and new kernels at each budget and checks all returned box coordinates, five keypoints and scores.

## Reproduce

Run in a git checkout containing the reference revision cb672bee, with GCC, NumPy, Pillow and SciPy installed.

```sh
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_detail.py yunet.bin WIDER_val.zip wider_face_split.zip
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_kernels.py yunet.bin WIDER_val.zip
```

## Results and decision

The 320 evaluation images had 2677 valid annotations, including 2171 small faces. The guided pass was triggered on 125 images. On development images the current, full 800 and guided modes gave 17, 16 and 17 unmatched detections respectively.

| Mode | Score | Total matches | Small matches | Unmatched | Small recovered / lost vs current | Mean CPU ms, reference kernels |
|---|---:|---:|---:|---:|---:|---:|
| Current | 0.70 | 888 | 425 | 37 | 0 / 0 | 83.13 |
| Soft | 0.55 | 939 | 468 | 82 | 43 / 0 | 83.13 |
| Full 800 | 0.75 | 999 | 538 | 32 | 141 / 28 | 126.49 |
| Guided 448 | 0.76 | 977 | 517 | 25 | 116 / 24 | 98.50 |

Full 800 improved small recall from 19.6% to 24.8%; its gain was 113 small matches (+26.6%). The 95% paired interval was +65 to +171 small matches and −16 to +7 unmatched predictions. Guided refinement gained 92 small matches (interval +52 to +140), with 12 fewer unmatched predictions (interval −24 to −1). It remains a research mode. Lowering the threshold alone more than doubled the unmatched count. Small-face recall is still below 25% on this subset.

On 64 separately sampled WIDER frames (plus one warmup), AMD EPYC 9V74, GCC 14 native, median milliseconds for resize, conversion and detection:

| Canvas budget | Previous kernels | New kernels |
|---|---:|---:|
| 640² | 77.41 | 51.58 |
| 800² | 121.04 | 73.45 |

All returned scores, box coordinates and landmarks were bitwise equal for both builds on both sizes in this test. The local scalar build and the optimized native build passed the ONNX fixture. Timings exclude face embeddings and do not establish performance on other hardware. Relative to the optimized 640 path, the 800 mode costs about 42% more median CPU time. Both SDK and REST expose full 800 as an explicit choice; REST keeps the faster 640 policy as its default.
