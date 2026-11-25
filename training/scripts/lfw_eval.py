"""
LFW verification eval, InsightFace-style.

Expects an `lfw.bin` file (InsightFace format) at D:/apps/facex/training/data/lfw.bin
which is a pickle of:
    bins:    list of byte strings (jpeg/png), 12000 entries (6000 pairs * 2)
    issame:  list of 6000 bools

Outputs accuracy at the best threshold found via 10-fold CV.
"""

import io
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__:
    from .verification_thresholds import fold_metrics
else:
    from verification_thresholds import fold_metrics


def load_lfw(bin_path: str):
    with open(bin_path, "rb") as f:
        # InsightFace pickled with protocol that requires latin1 fallback
        try:
            bins, issame = pickle.load(f, encoding="bytes")
        except Exception:
            f.seek(0)
            bins, issame = pickle.load(f, encoding="latin1")
    return bins, issame


def decode_to_tensor(jpeg_bytes: bytes, size: int = 112) -> torch.Tensor:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 128.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


@torch.no_grad()
def evaluate(model, lfw_bin_path: str, device: str = "cuda",
             batch_size: int = 256) -> dict:
    bins, issame = load_lfw(lfw_bin_path)
    n_imgs = len(bins)
    n_pairs = len(issame)
    assert n_imgs == 2 * n_pairs

    model.eval()
    embs = []
    for start in range(0, n_imgs, batch_size):
        chunk = bins[start:start + batch_size]
        # also embed the horizontally flipped version, average — standard trick
        x = torch.stack([decode_to_tensor(b) for b in chunk]).to(device)
        e1 = model(x)
        e2 = model(torch.flip(x, dims=[3]))
        e = F.normalize(e1 + e2, dim=-1)
        embs.append(e.cpu())
    embs = torch.cat(embs, dim=0)            # [n_imgs, D]
    a = embs[0::2]
    b = embs[1::2]
    sims = (a * b).sum(dim=-1).numpy()        # cosine, [n_pairs]
    issame = np.asarray(issame, dtype=bool)

    accs, best_thresholds = fold_metrics(sims, issame, np.linspace(0.0, 1.0, 401), 10)
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "threshold_mean": float(np.mean(best_thresholds)),
        "n_pairs": n_pairs,
    }
