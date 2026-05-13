"""
Sequentially train all 4 FaceX models with auto-resume.

Order is small -> large so the smallest models (which are fastest) finish
first and give early validation that the pipeline works end-to-end.

Each model:
  - trains for --epochs (defaults below per arch)
  - resumes from runs/<arch>/last.pt if present
  - exports best checkpoint to weights/facex_<arch>.bin

Usage:
    python train_all.py
    python train_all.py --only nano,tiny
    python train_all.py --skip xs
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import arch as arch_module

ROOT = Path("D:/apps/facex/training")
SCRIPTS = ROOT / "scripts"
RUNS = ROOT / "runs"
WEIGHTS_OUT = Path("D:/apps/facex/weights")

# Per-arch defaults. With LayerScale gamma=1e-6, training is slow but
# stable. Need many epochs for loss to drop into the useful range.
DEFAULTS = {
    "nano":     dict(epochs=40, batch=512, lr=1e-3),
    "tiny":     dict(epochs=35, batch=384, lr=1e-3),
    "standard": dict(epochs=30, batch=256, lr=1e-3),
    "xs":       dict(epochs=30, batch=192, lr=1e-3),
}

# Train XS first — 1.78M params is large enough to confirm the recipe
# converges. If XS hits a real LFW number we know nano/tiny/standard
# just need their own runs (or distillation from XS).
ORDER = ["xs", "standard", "tiny", "nano"]


def run_train(name: str):
    cfg = DEFAULTS[name]
    cmd = [sys.executable, str(SCRIPTS / "train.py"),
           "--arch", name,
           "--epochs", str(cfg["epochs"]),
           "--batch", str(cfg["batch"]),
           "--lr", str(cfg["lr"]),
           "--resume"]
    print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
    print("> " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(SCRIPTS))
    dt = time.time() - t0
    print(f"\n[{name}] training process exited code {p.returncode} after {dt/3600:.2f}h")
    return p.returncode == 0


def export(name: str):
    src = RUNS / name / "best.pt"
    if not src.exists():
        # fall back to last.pt if best.pt doesn't exist (LFW eval may have been skipped)
        src = RUNS / name / "last.pt"
    if not src.exists():
        print(f"[{name}] no checkpoint found, skipping export")
        return False
    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    out = WEIGHTS_OUT / f"facex_{name}.bin"
    cmd = [sys.executable, str(SCRIPTS / "export_bin.py"),
           "--arch", name, "--ckpt", str(src), "--out", str(out)]
    print("> " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(SCRIPTS))
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated arch names")
    ap.add_argument("--skip", help="comma-separated arch names to skip")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    targets = ORDER[:]
    if args.only:
        only = set(args.only.split(","))
        targets = [n for n in targets if n in only]
    if args.skip:
        skip = set(args.skip.split(","))
        targets = [n for n in targets if n not in skip]

    print(f"Will train: {targets}")
    results = {}
    for name in targets:
        ok = run_train(name)
        if ok and not args.no_export:
            export(name)
        results[name] = ok

    print("\n" + "=" * 60)
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
