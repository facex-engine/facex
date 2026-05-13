"""Train tongue-out binary classifier."""

import argparse, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import TongueNet
from dataset import TongueDataset


def cosine_lr(step, total, warmup, base, mn):
    if step < warmup: return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}")

    full = TongueDataset(augment=True)
    n = len(full); nt = int(n * 0.9); ne = n - nt
    train_ds, test_ds = random_split(full, [nt, ne],
                                      generator=torch.Generator().manual_seed(0))
    print(f"train {nt:,}  test {ne:,}")

    train_l = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True,
                         persistent_workers=(args.workers > 0), drop_last=True)
    test_l = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = TongueNet().to(device)
    print(f"params: {model.num_params():,}")

    # Plain cross-entropy (no class weights) — earlier 4.6x positive weight
    # made the model over-fire on every face. With ~1:4.6 ratio CE alone
    # converges fine and gives much cleaner probabilities.
    class_weights = None

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if p.ndim < 2 or n_.endswith(".bias") or "bn" in n_.lower(): no_decay.append(p)
        else: decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    runs = Path("D:/apps/facex/training/tongue/runs"); runs.mkdir(parents=True, exist_ok=True)
    log_f = open(runs / "train.log", "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== run: ep={args.epochs} batch={args.batch} ===\n")
    best = 0.0
    spe = len(train_l); total_steps = spe * args.epochs; gs = 0

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); n_seen = 0; loss_acc = 0
        for it, (imgs, lbl) in enumerate(train_l):
            imgs = imgs.to(device, non_blocking=True); lbl = lbl.to(device, non_blocking=True)
            lr = cosine_lr(gs, total_steps, args.warmup, args.lr, args.min_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs)
                loss = F.cross_entropy(logits, lbl, weight=class_weights) if class_weights is not None else F.cross_entropy(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n_seen += imgs.size(0); gs += 1; loss_acc += loss.item()

        # eval
        model.eval(); tp=0; fp=0; tn=0; fn=0
        with torch.no_grad():
            for imgs, lbl in test_l:
                imgs = imgs.to(device); lbl = lbl.to(device)
                pred = model(imgs).argmax(-1)
                tp += ((pred == 1) & (lbl == 1)).sum().item()
                fp += ((pred == 1) & (lbl == 0)).sum().item()
                tn += ((pred == 0) & (lbl == 0)).sum().item()
                fn += ((pred == 0) & (lbl == 1)).sum().item()
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        msg = (f"ep {ep+1}/{args.epochs}  acc {acc:.3f}  prec {prec:.3f}  rec {rec:.3f}  "
               f"f1 {f1:.3f}  loss {loss_acc/spe:.3f}  time {(time.time()-t0)/60:.1f}min  lr {lr:.2e}")
        print(msg, flush=True); log_f.write(msg + "\n")
        torch.save({"epoch": ep, "model": model.state_dict()}, runs / "last.pt")
        if f1 > best:
            best = f1
            torch.save({"epoch": ep, "model": model.state_dict(), "f1": f1,
                        "prec": prec, "rec": rec, "acc": acc}, runs / "best.pt")
            log_f.write(f"  -> new best F1 {f1:.3f}\n")

    log_f.close()
    print(f"done. best F1 {best:.3f}")


if __name__ == "__main__":
    main()
