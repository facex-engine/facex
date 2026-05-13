"""
MS1M-pair verification eval (LFW substitute when LFW.bin unavailable).

Samples N same-person and N different-person pairs from MS1M, runs each
trained model, computes cosine similarities, finds optimal threshold
via 10-fold cross-validation, and reports verification accuracy.

NOTE: MS1M identities were seen during training, so these numbers are
inflated vs. true unseen-identity verification (LFW). They are still a
useful sanity check — a model that scores low here would never pass
LFW. A model that scores high here gives a *ceiling* estimate.

Usage:
    python eval_pairs.py --arch standard --ckpt ../runs/standard/last.pt
    python eval_pairs.py --all
"""

import argparse
import io
import struct
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import arch as arch_module
from model import MobileFaceNet
from dataset import MS1MDataset

DATA_ROOT = Path("D:/apps/facex/training/data/ms1m")
INDEX = struct.Struct("<QII")
SEED = 1337
N_PAIRS_PER_TYPE = 3000        # 3000 same + 3000 diff = 6000 pairs total
N_FOLDS = 10


def build_pairs(seed: int = SEED, n_per: int = N_PAIRS_PER_TYPE):
    """Build a balanced list of (img_idx_a, img_idx_b, same_label) pairs."""
    rng = np.random.default_rng(seed)
    # We need quick lookup of all images per label. Walk the index once.
    print("Building label -> images map...")
    blob_path = DATA_ROOT / "blob.jpg.bin"
    idx_path = DATA_ROOT / "index.bin"
    idx_mm = np.memmap(idx_path, dtype=np.uint8, mode="r")
    n = idx_mm.size // 16
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        _, _, lbl = INDEX.unpack(idx_mm[i * 16:(i + 1) * 16].tobytes())
        labels[i] = lbl

    # Bucket images by label
    by_label = {}
    for i, lb in enumerate(labels.tolist()):
        by_label.setdefault(lb, []).append(i)
    multi = [lb for lb, ix in by_label.items() if len(ix) >= 2]
    print(f"{n:,} images, {len(by_label):,} classes, {len(multi):,} with >=2 imgs")

    pairs = []
    # same-pairs
    same_classes = rng.choice(multi, size=n_per, replace=True)
    for cls in same_classes:
        ids = by_label[int(cls)]
        i, j = rng.choice(ids, size=2, replace=False)
        pairs.append((int(i), int(j), True))
    # diff-pairs
    for _ in range(n_per):
        c1, c2 = rng.choice(multi, size=2, replace=False)
        i = int(rng.choice(by_label[int(c1)]))
        j = int(rng.choice(by_label[int(c2)]))
        pairs.append((i, j, False))
    rng.shuffle(pairs)
    print(f"Built {len(pairs):,} pairs ({n_per:,} same / {n_per:,} diff)")
    return pairs


def load_image_chw(ds, idx, size: int = 112):
    img, _ = ds[idx]
    return img  # already CHW float [-1,1]


@torch.no_grad()
def compute_embeddings_for_pairs(model, pairs, ds, device="cuda",
                                  batch_size: int = 128):
    """Returns (sims [N], issame [N])."""
    model.eval()
    n = len(pairs)
    # Unique image indices to embed (each pair has 2)
    all_idxs = sorted({i for p in pairs for i in (p[0], p[1])})
    idx_to_pos = {i: k for k, i in enumerate(all_idxs)}
    M = len(all_idxs)
    embs = []
    t0 = time.time()
    for start in range(0, M, batch_size):
        chunk = all_idxs[start:start + batch_size]
        xs = torch.stack([load_image_chw(ds, i) for i in chunk]).to(device)
        e1 = model(xs)
        e2 = model(torch.flip(xs, dims=[3]))
        e = torch.nn.functional.normalize(e1 + e2, dim=-1)
        embs.append(e.cpu().numpy())
        if (start // batch_size) % 10 == 0:
            dt = time.time() - t0
            print(f"  embedded {start + len(chunk)}/{M}  ({(start+len(chunk))/dt:.0f} img/s)")
    E = np.concatenate(embs, axis=0)  # [M, D]
    sims = np.zeros(n, dtype=np.float32)
    issame = np.zeros(n, dtype=bool)
    for k, (a, b, same) in enumerate(pairs):
        sims[k] = float(np.dot(E[idx_to_pos[a]], E[idx_to_pos[b]]))
        issame[k] = same
    return sims, issame


def kfold_threshold(sims: np.ndarray, issame: np.ndarray, n_folds: int = N_FOLDS):
    """10-fold CV: find best threshold on train folds, evaluate on held-out."""
    n = len(sims)
    folds = np.array_split(np.arange(n), n_folds)
    thrs = np.linspace(-0.5, 1.0, 601)
    accs, best_thrs = [], []
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        best_acc, best_thr = 0.0, 0.0
        for thr in thrs:
            pred = sims[train_idx] >= thr
            a = (pred == issame[train_idx]).mean()
            if a > best_acc:
                best_acc, best_thr = a, thr
        pred = sims[test_idx] >= best_thr
        accs.append((pred == issame[test_idx]).mean())
        best_thrs.append(best_thr)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "thr_mean": float(np.mean(best_thrs)),
    }


def eval_one(arch_name: str, ckpt_path: str, pairs, ds, device: str):
    arch = arch_module.ALL_ARCHS[arch_name]
    model = MobileFaceNet(arch).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    sims, issame = compute_embeddings_for_pairs(model, pairs, ds, device=device)
    # Quick separation stats
    same_sims = sims[issame]
    diff_sims = sims[~issame]
    print(f"  same: mean={same_sims.mean():.4f}  std={same_sims.std():.4f}")
    print(f"  diff: mean={diff_sims.mean():.4f}  std={diff_sims.std():.4f}")
    print(f"  separation gap: {same_sims.mean() - diff_sims.mean():.4f}")
    stats = kfold_threshold(sims, issame)
    print(f"  verification accuracy: {stats['acc_mean']*100:.2f}% "
          f"+/- {stats['acc_std']*100:.2f}%  (thr={stats['thr_mean']:.3f})")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(arch_module.ALL_ARCHS.keys()))
    ap.add_argument("--ckpt")
    ap.add_argument("--all", action="store_true",
                    help="Eval all 4 architectures from their last.pt")
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS_PER_TYPE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MS1MDataset(DATA_ROOT, augment=False)
    pairs = build_pairs(n_per=args.n_pairs)

    targets = []
    if args.all:
        for n in ["nano", "tiny", "standard", "xs"]:
            p = Path(f"D:/apps/facex/training/runs/{n}/last.pt")
            if p.exists():
                targets.append((n, str(p)))
    else:
        if not args.arch or not args.ckpt:
            raise SystemExit("either --all or both --arch and --ckpt")
        targets.append((args.arch, args.ckpt))

    print()
    results = {}
    for name, ckpt in targets:
        print(f"=== {name}  ({ckpt}) ===")
        results[name] = eval_one(name, ckpt, pairs, ds, device)
        print()

    print("=" * 60)
    print(f"{'arch':>10}  {'verify_acc':>12}  {'std':>8}  {'thr':>8}")
    print("-" * 60)
    for n, s in results.items():
        print(f"{n:>10}  {s['acc_mean']*100:>10.2f}%  {s['acc_std']*100:>6.2f}%  {s['thr_mean']:>8.3f}")


if __name__ == "__main__":
    main()
