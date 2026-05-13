"""
ArcFace training for one of the 4 FaceX architectures.

Usage:
    python train.py --arch nano --epochs 25 --batch 256
    python train.py --arch xs --epochs 25 --batch 192 --resume

Checkpoints go to D:/apps/facex/training/runs/<arch>/
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import arch as arch_module
from model import FaceXModel, ArcFaceHead
from dataset import MS1MDataset
from lfw_eval import evaluate as lfw_eval

DATA_ROOT = Path("D:/apps/facex/training/data/ms1m")
LFW_BIN = Path("D:/apps/facex/training/data/lfw.bin")
RUNS_ROOT = Path("D:/apps/facex/training/runs")


def build_optimizer(model: nn.Module, head: nn.Module, lr: float, wd: float):
    # SGD+momentum is the canonical recipe for ArcFace face recognition
    # (InsightFace's arcface_torch uses it). AdamW@1e-3 plateaus around
    # loss=22 on MS1M with 85K classes; SGD@0.1 breaks past that easily.
    # Skip weight-decay on biases, LayerNorm scales/biases, gammas, temperature.
    decay, no_decay = [], []
    for name, p in list(model.named_parameters()) + [(f"head.{n}", q)
                                                     for n, q in head.named_parameters()]:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "gamma" in name or "ln" in name.lower() or "temperature" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.999),
    )


def cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float, min_lr: float):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(arch_module.ALL_ARCHS.keys()))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arc-s", type=float, default=64.0)
    ap.add_argument("--arc-m", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1, help="epochs between LFW evals")
    args = ap.parse_args()

    arch = arch_module.ALL_ARCHS[args.arch]
    parts = arch_module.count_params(arch)
    print(f"Arch: {arch.name}  params={parts['TOTAL']:,}  emb={arch.embedding_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")

    # data
    ds = MS1MDataset(DATA_ROOT, augment=True)
    print(f"Dataset: {len(ds):,} images, {ds.n_classes:,} classes")
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=(args.workers > 0), drop_last=True,
    )

    # model + head
    model = FaceXModel(arch).to(device)
    head = ArcFaceHead(arch.embedding_dim, ds.n_classes, s=args.arc_s, m=args.arc_m).to(device)
    print(f"Model parameter count (incl. head): "
          f"{sum(p.numel() for p in model.parameters()):,} backbone + "
          f"{sum(p.numel() for p in head.parameters()):,} head")

    opt = build_optimizer(model, head, args.lr, args.wd)

    run_dir = RUNS_ROOT / arch.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    ckpt_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    start_epoch = 0
    best_acc = 0.0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    global_step = start_epoch * steps_per_epoch

    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== run start: arch={arch.name} epochs={args.epochs} batch={args.batch} ===\n")

    for epoch in range(start_epoch, args.epochs):
        model.train(); head.train()
        t0 = time.time()
        loss_sum, acc_sum, n_seen = 0.0, 0.0, 0
        for it, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, args.warmup_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            # Run fully in fp32 — bf16 autocast on this small a network at
            # 85K classes loses too much gradient signal and stalls training
            # around target_cos ~= 0.33. The throughput hit is ~30%.
            emb = model(imgs)
            logits = head(emb, labels)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            loss_sum += loss.item() * imgs.size(0)
            acc_sum += acc * imgs.size(0)
            n_seen += imgs.size(0)
            global_step += 1

            if (it + 1) % 100 == 0:
                avg_loss = loss_sum / n_seen
                avg_acc = acc_sum / n_seen
                ips = n_seen / (time.time() - t0)
                msg = (f"epoch {epoch+1}/{args.epochs}  it {it+1}/{steps_per_epoch}  "
                       f"loss {avg_loss:.3f}  acc {avg_acc:.3f}  lr {lr:.2e}  "
                       f"{ips:.0f} img/s")
                print(msg)
                log_f.write(msg + "\n")

        epoch_dt = time.time() - t0
        msg = (f"[epoch done] epoch {epoch+1} loss {loss_sum/n_seen:.3f} "
               f"acc {acc_sum/n_seen:.3f} time {epoch_dt/60:.1f} min")
        print(msg); log_f.write(msg + "\n")

        # save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "head": head.state_dict(),
            "opt": opt.state_dict(),
            "best_acc": best_acc,
            "arch_name": arch.name,
        }, ckpt_path)

        # eval
        if (epoch + 1) % args.eval_every == 0 and LFW_BIN.exists():
            try:
                stats = lfw_eval(model, str(LFW_BIN), device=device)
                msg = (f"[lfw] epoch {epoch+1} acc={stats['accuracy_mean']:.4f} "
                       f"+/- {stats['accuracy_std']:.4f} thr={stats['threshold_mean']:.3f}")
                print(msg); log_f.write(msg + "\n")
                if stats["accuracy_mean"] > best_acc:
                    best_acc = stats["accuracy_mean"]
                    torch.save({"model": model.state_dict(),
                                "arch_name": arch.name,
                                "epoch": epoch,
                                "lfw_acc": best_acc}, best_path)
                    log_f.write(f"[lfw] new best -> {best_path}\n")
            except Exception as e:
                msg = f"[lfw] eval failed: {e}"
                print(msg); log_f.write(msg + "\n")

    log_f.close()
    print(f"Training complete. Best LFW acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
