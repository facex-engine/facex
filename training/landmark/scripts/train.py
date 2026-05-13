"""
Train the WFLW landmark model.

Usage:
    python train.py --epochs 60 --batch 256
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import LandmarkNet, WingLoss
from dataset import WFLWDataset

DATA_ROOT = Path("D:/apps/facex/training/data/wflw")
RUNS_ROOT = Path("D:/apps/facex/training/landmark/runs")


def normalized_mean_error(pred, target):
    """Inter-ocular NME — distance between eye corners (96, 97) as reference."""
    P = pred.view(-1, 98, 2)
    T = target.view(-1, 98, 2)
    # WFLW: 96 = pupil-left eye, 97 = pupil-right eye
    eye_dist = torch.norm(T[:, 96] - T[:, 97], dim=-1)
    nme_per_pt = torch.norm(P - T, dim=-1)
    nme = nme_per_pt.mean(dim=-1) / (eye_dist + 1e-6)
    return nme.mean().item()


def cosine_lr(step, total, warmup, base, mn):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = WFLWDataset(DATA_ROOT / "train", augment=True)
    test_ds = WFLWDataset(DATA_ROOT / "test", augment=False)
    print(f"train: {len(train_ds):,}  test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0), drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = LandmarkNet().to(device)
    print(f"params: {model.num_params():,}")
    crit = WingLoss(w=10.0, eps=2.0)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or n.endswith(".bias") or "bn" in n.lower() or "act" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=args.lr)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt = RUNS_ROOT / "last.pt"
    best = RUNS_ROOT / "best.pt"
    log_f = open(RUNS_ROOT / "train.log", "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== run start: epochs={args.epochs} batch={args.batch} ===\n")

    start_epoch = 0
    best_nme = float("inf")
    if args.resume and ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        start_epoch = c["epoch"] + 1
        best_nme = c.get("best_nme", float("inf"))
        print(f"resumed from epoch {start_epoch}, best_nme={best_nme:.4f}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0; n_seen = 0
        for it, (imgs, pts) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            pts = pts.to(device, non_blocking=True)
            lr = cosine_lr(global_step, total_steps, args.warmup, args.lr, args.min_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            pred = model(imgs)
            loss = crit(pred, pts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item() * imgs.size(0); n_seen += imgs.size(0)
            global_step += 1

        # eval
        model.eval()
        nmes = []
        with torch.no_grad():
            for imgs, pts in test_loader:
                imgs = imgs.to(device); pts = pts.to(device)
                pred = model(imgs)
                nmes.append(normalized_mean_error(pred, pts))
        nme = float(np.mean(nmes))
        msg = (f"epoch {epoch+1}/{args.epochs}  train_loss={loss_sum/n_seen:.3f}  "
               f"test_NME={nme*100:.3f}%  time={(time.time()-t0)/60:.1f} min  lr={lr:.2e}")
        print(msg); log_f.write(msg + "\n")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "opt": opt.state_dict(), "best_nme": best_nme}, ckpt)
        if nme < best_nme:
            best_nme = nme
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "nme": nme}, best)
            log_f.write(f"  -> new best {nme*100:.3f}%\n")

    log_f.close()
    print(f"done. best test NME: {best_nme*100:.3f}%")


if __name__ == "__main__":
    main()
