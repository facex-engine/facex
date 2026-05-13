# FaceX training — pause/resume guide

## Current state (saved)

| arch | params | status | last checkpoint | last epoch | acc |
|------|-------:|--------|-----------------|-----------:|----:|
| standard | 986K | DONE 30/30 | `runs/standard/last.pt` (538 MB) | 30 | 38.2% |
| xs | 2.1M | paused | `runs/xs/last.pt` (552 MB) | 2 (epoch 3 in-progress) | 17.7% |
| tiny | 464K | not started | — | — | — |
| nano | 198K | not started | — | — | — |

Exported weights:
- `weights/facex_standard.bin` (3.95 MB FP32 / ≈1 MB INT8)

## How to resume

All scripts are in `training/scripts/`. The dataset, model, ArcFace head,
and recipe are committed in code; nothing else to set up.

### Resume XS (will restart from epoch 3, since it saved checkpoint at end of epoch 2)

```bash
cd training/scripts
python train_all.py --only xs
```

The `--resume` flag is on by default in train_all. It detects `last.pt`
and continues from there with optimizer state restored.

### Train remaining models (Tiny, Nano)

```bash
# After XS finishes:
python train_all.py --only tiny
python train_all.py --only nano

# Or all three at once (sequential):
python train_all.py --only tiny,nano
```

`train_all.py` auto-exports each finished model to
`weights/facex_<arch>.bin`.

### Train all from scratch (clean start)

```bash
rm -rf D:/apps/facex/training/runs/*
python train_all.py
```

### Resume after a crash (any model)

train.py saves `last.pt` after every epoch with model + head + optimizer
state. `--resume` (default in train_all) picks up exactly where it left
off, with the cosine LR schedule continuing.

## Realistic time estimates (5060 Ti)

| arch | min/epoch | total (30 ep) |
|------|----------:|--------------:|
| nano | ~30 | ~15 h |
| tiny | ~50 | ~25 h |
| standard | 80 | 40 h ✓ done |
| xs | ~128 | ~64 h |

## Trajectory snapshot for XS so far

```
epoch 1 loss 21.438 acc 0.058 time 129.0 min
epoch 2 loss 11.206 acc 0.177 time 127.9 min
epoch 3 in-progress — last seen iter 10300/30326, loss 9.58, acc 23.2%
```

XS converging faster than standard did (e.g. acc 17.7% at ep 2 vs 12.7%
for standard). Expected final acc on MS1M: ~40-45%.

## Note on accuracy interpretation

MS1M classification accuracy of 38% looks low because the dataset has
85,742 classes (random = 0.001%). Verification accuracy (LFW) will be
much higher — typical conversion is 38% MS1M classify -> ~95-97% LFW.

To run LFW eval after training: drop an InsightFace-format `lfw.bin`
into `training/data/lfw.bin` and the per-epoch eval will start running.
The eval code is in `lfw_eval.py`.

## Next session: where to start

1. Run `python train_all.py --only xs` to resume XS (will pick up epoch 3)
2. After XS finishes: `python train_all.py --only tiny,nano`
3. Verify all 4 .bin files are in `D:/apps/facex/weights/`
4. (Optional) write the C engine v3 for MobileFaceNet so the .bin files
   can be loaded by the FaceX runtime — see notes in
   `src/edgeface_engine_v2.c` for reference layout (the v2 engine handles
   the EdgeFace topology, but MobileFaceNet's BN+PReLU+bottleneck graph
   needs its own engine file)
