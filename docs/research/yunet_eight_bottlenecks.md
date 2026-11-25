# Eight bottlenecks in small face detection

The reference is the rectangular RGB YuNet path from cb672bee with score 0.58, NMS IoU 0.40 and up to 32 detections. Each experiment changes one element plus an independently calibrated score threshold. The recognition network and pixel budget stay fixed.

| Trial | Mechanism | Candidate change |
|---|---|---|
| Minimum side | A candidate disappears when either predicted side falls below the hard minimum | Lower the cutoff from 16 to 8 canvas pixels |
| Aspect ratio | Strongly elongated boxes fail the aspect filter | Widen the accepted width/height interval from [0.30, 3.0] to [0.18, 4.0] |
| Box suppression | Two nearby faces can suppress each other through IoU | Raise the suppression threshold from 0.40 to 0.55 |
| Landmark suppression | Fixed pixel distances merge nearby people regardless of face size | Scale nose and eye midpoint distances to 0.25 and 0.20 times the smaller geometric mean box side |
| Output limit | Dense scenes can exceed the 32 result slots | Admit 128 sorted boxes |
| Scale dependent confidence | A common threshold can disadvantage small boxes | Multiply score by 1.10 for candidates with a side below 32 pixels, cap at one before ranking |
| Box calibration | Correct centers may still miss IoU 0.5 because of size bias | Expand each side outwards by 4% of width or height after suppression |
| Channel order | First layer weights may respond differently to channel order | Supply BGR to the detector while retaining the original RGB image for alignment |

## Evaluation

WIDER FACE validation. Exclude the 640 evaluation images and 128 calibration images used in the rectangular input experiment. Draw 672 other images without replacement with seed 3519. The first 160 are for calibration and the remaining 512 for evaluation. This split isolates the eight-way comparison within this experiment. Images come from the same validation dataset.

For each variant select the score threshold on a 0.01 grid from 0.40 to 0.91 to maximize small matches under the baseline calibration false-positive count. Break ties using total matches and then proximity to 0.58. Select one primary variant by calibration small matches before scoring the evaluation images. Report all eight with frozen thresholds. The baseline uses 0.58 throughout.

The reference C head emits all positive-size candidates with score at least 0.30 in the experimental build. Replay uses stable descending score order followed by the native geometry filters, greedy box suppression, landmark suppression and output cap. Every thirteenth image is checked against the unmodified native detector. Boxes are mapped and clipped to original coordinates. One-to-one IoU ≥ 0.50 matching uses valid annotations. Unmatched detections include predictions on ignored annotations. A small face has geometric mean side below 16 pixels after scaling the long side of the original image to 320. These are fixed operating point counts. They are not official WIDER AP.

## Reproduction

Run from a checkout containing the reference commit, with GCC, NumPy, SciPy, Pillow, native YuNet 2023mar weights and both WIDER validation archives.

```sh
OPENBLAS_NUM_THREADS=1 python tests/bench_yunet_bottlenecks.py yunet.bin WIDER_val.zip wider_face_split.zip
```

## Results on the 512 evaluation images

There are 5173 valid faces including 4328 small faces. Thresholds were calibrated using 160 other images; the reference calibration count was 31 unmatched predictions. Changing the threshold along with each variant measures its deployable operating point.

| Trial | Score | Total matches | Small matches | Unmatched |
|---|---:|---:|---:|---:|
| Reference | 0.58 | 1306 | 686 | 94 |
| Minimum side 8 | 0.62 | 1317 | 703 | 80 |
| Wider aspect range | 0.58 | 1306 | 686 | 94 |
| NMS IoU 0.55 | 0.58 | 1306 | 686 | 95 |
| Relative landmark distances | 0.59 | 1313 | 691 | 95 |
| Result limit 128 | 0.59 | 1300 | 681 | 85 |
| Small box score bonus | 0.63 | 1295 | 685 | 93 |
| Box expansion | 0.60 | 1281 | 670 | 89 |
| Detector BGR | 0.66 | 1728 | 996 | 106 |

BGR was selected on calibration images before evaluation. At 0.66 the paired 95% image bootstrap interval was +248 to +374 small matches and −22 to +46 unmatched predictions relative to RGB. Lowering the minimum size gave +17 small matches with a paired interval of −4 to +42. The cap trial includes a threshold increase from 0.58 to 0.59 to respect the calibration budget; the comparison does not imply that a larger output buffer by itself loses detections.

On the RGB reference, an oracle that discards every incorrect candidate and optimally assigns the remainder to small annotations could match at most 1066 of 4328 at score ≥0.30, before any suppression or geometry filters. At score ≥0.58 that ceiling was 760; a minimum side of 8 gave 760 and a minimum side of 16 gave 706. This ceiling applies only to the RGB candidate boxes, channel order, score floor and IoU criterion tested.

## Independent color confirmation and release decision

The first comparison gained matches but showed 12 additional unmatched predictions. For the REST operating point we tightened the original calibration budget to 75% of baseline (at most 23 unmatched predictions on the same 160 calibration images) and selected 0.70. We then evaluated 384 further images, drawn with seed 9101 outside all the earlier splits. There were 4114 valid faces including 3411 small faces.

| Detector input and score | Total matches | Small matches | Unmatched |
|---|---:|---:|---:|
| RGB 0.58 | 1061 | 548 | 58 |
| BGR 0.58 | 1399 | 776 | 109 |
| BGR 0.66 | 1365 | 749 | 76 |
| BGR 0.70 | 1344 | 734 | 56 |

At 0.70 the gain was 186 small faces (+33.9%, recall 16.1% to 21.5%). The paired 95% image bootstrap intervals were +129 to +255 small matches, +222 to +358 total matches and −22 to +19 unmatched predictions. The observed difference of two fewer unmatched predictions is within sampling uncertainty. A separate alternating 48-image, single-core Linux CPU run of preprocessing plus native detection measured median 75.31 ms for RGB and 76.22 ms for BGR; neither embedding nor JPEG decoding was timed.

REST selects BGR 0.70 for the detector only. Input to the SDK, source coordinates, alignment and embedding use the original RGB image. Existing SDK policies retain their original thresholds and behavior. API integration was checked on landscape, portrait and square images against a direct native call with explicit channel reversal, including all embedding components.

```sh
OPENBLAS_NUM_THREADS=1 python tests/verify_yunet_color.py yunet.bin WIDER_val.zip wider_face_split.zip
```
