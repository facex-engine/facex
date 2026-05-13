"""Train binary anti-spoof classifier."""

import argparse, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import AntiSpoofNet
from dataset import AntiSpoofDataset


def cosine_lr(step, total, warmup, base, mn):
    if step < warmup: return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    full = AntiSpoofDataset(augment=True)
    n = len(full); nt = int(n * 0.95); ne = n - nt
    train_ds, test_ds = random_split(full, [nt, ne], generator=torch.Generator().manual_seed(0))
    print(f"train {nt:,}  test {ne:,}")

    train_l = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True,
                         persistent_workers=(args.workers > 0), drop_last=True)
    test_l = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = AntiSpoofNet().to(device)
    print(f"params: {model.num_params():,}")
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if p.ndim < 2 or n_.endswith(".bias") or "bn" in n_.lower() or "act" in n_.lower():
            no_decay.append(p)
        else: decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    runs = Path("D:/apps/facex/training/antispoof/runs"); runs.mkdir(parents=True, exist_ok=True)
    log_f = open(runs / "train.log", "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== run: ep={args.epochs} batch={args.batch} ===\n")
    best = float("inf")
    spe = len(train_l); total = spe * args.epochs; gs = 0

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); n_seen = 0
        for it, (imgs, lbl) in enumerate(train_l):
            imgs = imgs.to(device, non_blocking=True); lbl = lbl.to(device, non_blocking=True)
            lr = cosine_lr(gs, total, args.warmup, args.lr, args.min_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs)
                loss = F.cross_entropy(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n_seen += imgs.size(0); gs += 1
            if (it + 1) % 50 == 0:
                msg = f"  ep {ep+1} it {it+1}/{spe} loss {loss.item():.3f} {n_seen/(time.time()-t0):.0f} img/s lr {lr:.2e}"
                print(msg, flush=True); log_f.write(msg + "\n")

        # eval
        model.eval(); correct = 0; total_n = 0
        with torch.no_grad():
            for imgs, lbl in test_l:
                imgs = imgs.to(device); lbl = lbl.to(device)
                logits = model(imgs)
                correct += (logits.argmax(-1) == lbl).sum().item()
                total_n += lbl.size(0)
        acc = correct / total_n
        err = 1 - acc
        msg = f"ep {ep+1}/{args.epochs}  test_acc {acc*100:.2f}%  err {err*100:.2f}%  time {(time.time()-t0)/60:.1f}min  lr {lr:.2e}"
        print(msg, flush=True); log_f.write(msg + "\n")
        torch.save({"epoch": ep, "model": model.state_dict()}, runs / "last.pt")
        if err < best:
            best = err
            torch.save({"epoch": ep, "model": model.state_dict(), "acc": acc}, runs / "best.pt")
            log_f.write(f"  -> new best acc {acc*100:.2f}%\n")

    log_f.close()
    print(f"done. best test_acc {(1-best)*100:.2f}%")


if __name__ == "__main__":
    main()
