"""Train the 3D distillation model."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model import Landmark3DNet, DistillLoss
from dataset import MP3DDataset


def cosine_lr(step, total, warmup, base, mn):
    if step < warmup: return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default="", help="path to .pt with model weights (no optimizer/epoch)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    full = MP3DDataset(augment=True)    # perspective aug enables side-view poses
    print(f"total samples: {len(full):,}")
    # 95/5 split
    n_train = int(len(full) * 0.95)
    n_test = len(full) - n_train
    gen = torch.Generator().manual_seed(0)
    train_ds, test_ds = random_split(full, [n_train, n_test], generator=gen)
    # disable augment on test
    test_ds_inner = MP3DDataset(augment=False)
    print(f"train: {len(train_ds):,}  test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0), drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = Landmark3DNet().to(device)
    print(f"params: {model.num_params():,}  (~{model.num_params()/1024:.0f} KB INT8)")
    crit = DistillLoss(w_xy=1.0, w_z=0.5)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or n.endswith(".bias") or "bn" in n.lower() or "act" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=args.lr)

    runs = Path("D:/apps/facex/training/landmark3d/runs"); runs.mkdir(parents=True, exist_ok=True)
    ckpt = runs / "last.pt"; best = runs / "best.pt"
    log_f = open(runs / "train.log", "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== run start: epochs={args.epochs} batch={args.batch} ===\n")

    start_epoch = 0; best_err = float("inf")
    if args.resume and ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        start_epoch = c["epoch"] + 1; best_err = c.get("best_err", float("inf"))
        print(f"resumed @ ep {start_epoch}, best_err={best_err:.3f}")
    elif args.init_from:
        c = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(c["model"] if "model" in c else c)
        print(f"loaded model weights from {args.init_from} (fresh optimizer)")

    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * args.epochs
    gs = start_epoch * steps_per_epoch

    import sys
    for epoch in range(start_epoch, args.epochs):
        model.train(); t0 = time.time(); loss_sum = 0; n_seen = 0
        for it, (imgs, pts) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True); pts = pts.to(device, non_blocking=True)
            lr = cosine_lr(gs, total, args.warmup, args.lr, args.min_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(imgs)
                loss = crit(pred, pts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n_seen += imgs.size(0)
            gs += 1
            if (it + 1) % 50 == 0:
                # sync once for the log (rare → not a bottleneck)
                dt = time.time() - t0
                l = loss.item()
                msg = f"  ep {epoch+1} it {it+1}/{steps_per_epoch} loss {l:.3f} {n_seen/dt:.0f} img/s lr {lr:.2e}"
                print(msg, flush=True); log_f.write(msg + "\n")
                loss_sum += l * imgs.size(0)   # accumulate for end-of-epoch avg
            else:
                loss_sum += 0   # don't sync mid-batch
        # accurate average only at end of epoch
        loss_sum = loss.item() * n_seen / max(1, n_seen)

        # eval
        model.eval(); errs_xy = []; errs_z = []
        with torch.no_grad():
            for imgs, pts in test_loader:
                imgs = imgs.to(device); pts = pts.to(device)
                pred = model(imgs)
                errs_xy.append(((pred[..., :2] - pts[..., :2]).abs().mean().item() * 112))
                errs_z.append(((pred[..., 2]  - pts[..., 2]).abs().mean().item() * 112))
        err_xy = float(np.mean(errs_xy)); err_z = float(np.mean(errs_z))
        msg = (f"ep {epoch+1}/{args.epochs}  trainL {loss_sum/n_seen:.3f}  "
               f"test_xy {err_xy:.2f}px  test_z {err_z:.2f}  "
               f"time {(time.time()-t0)/60:.1f}min  lr {lr:.2e}")
        print(msg); log_f.write(msg + "\n")

        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_err": best_err}, ckpt)
        combined = err_xy + 0.5 * err_z
        if combined < best_err:
            best_err = combined
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "err_xy": err_xy, "err_z": err_z}, best)
            log_f.write(f"  -> new best (combined {combined:.3f})\n")

    log_f.close()
    print(f"done. best combined err: {best_err:.3f}")


if __name__ == "__main__":
    main()
