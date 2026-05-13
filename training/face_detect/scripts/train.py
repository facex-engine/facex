"""Train the tiny face detector on WIDER FACE."""

import argparse, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import FaceDetector
from dataset import WiderFaceDataset, collate
from loss import DetectorLoss


def cosine_lr(step, total, warmup, base, mn):
    if step < warmup: return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--input-size", type=int, default=320)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}")

    train_ds = WiderFaceDataset("train", input_size=args.input_size, augment=True)
    val_ds   = WiderFaceDataset("val",   input_size=args.input_size, augment=False)

    train_l = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True,
                         persistent_workers=(args.workers > 0), drop_last=True,
                         collate_fn=collate)
    val_l = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True,
                       collate_fn=collate)

    model = FaceDetector().to(device)
    print(f"params: {model.num_params():,}")
    criterion = DetectorLoss(input_size=args.input_size).to(device)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or n.endswith(".bias") or "bn" in n.lower():
            no_decay.append(p)
        else: decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    runs = Path("D:/apps/facex/training/face_detect/runs"); runs.mkdir(parents=True, exist_ok=True)
    log_f = open(runs / "train.log", "a", buffering=1, encoding="utf-8")
    log_f.write(f"\n=== run: ep={args.epochs} batch={args.batch} lr={args.lr} ===\n")
    best = 0.0
    spe = len(train_l); total_steps = spe * args.epochs; gs = 0

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); n_seen = 0
        loss_acc = 0.0; cls_acc = 0.0; box_acc = 0.0; kps_acc = 0.0; n_pos_acc = 0
        for it, (imgs, boxes, kps) in enumerate(train_l):
            imgs = imgs.to(device, non_blocking=True)
            lr = cosine_lr(gs, total_steps, args.warmup, args.lr, args.min_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outs = model(imgs)
                loss, log = criterion(outs, boxes, kps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n_seen += imgs.size(0); gs += 1
            loss_acc += loss.item(); cls_acc += log["cls"]; box_acc += log["box"]
            kps_acc += log["kps"]; n_pos_acc += log["n_pos"]
            if (it + 1) % 50 == 0:
                dt = time.time() - t0
                msg = (f"  ep {ep+1} it {it+1}/{spe}  loss {loss_acc/(it+1):.3f}  "
                       f"cls {cls_acc/(it+1):.3f}  box {box_acc/(it+1):.3f}  "
                       f"kps {kps_acc/(it+1):.3f}  npos {n_pos_acc/(it+1):.0f}  "
                       f"{n_seen/dt:.0f} img/s  lr {lr:.2e}")
                print(msg, flush=True); log_f.write(msg + "\n")

        # quick eval: count positives at thresh=0.5 on val
        model.eval()
        n_pred = 0; n_gt = 0; n_match = 0
        with torch.no_grad():
            for imgs, boxes, kps in val_l:
                imgs = imgs.to(device)
                outs = model(imgs)
                # Decode + count rough TP@0.5 (mean per image)
                preds_per_img = decode_batch(outs, score_th=0.5)
                for pi, (pb, _) in enumerate(preds_per_img):
                    n_pred += len(pb)
                    gt = boxes[pi].to(device)
                    n_gt += len(gt)
                    if len(pb) and len(gt):
                        ious = box_iou(pb, gt)   # [P, G]
                        matched = ious.max(0).values > 0.5
                        n_match += matched.sum().item()
        recall = n_match / max(1, n_gt)
        precision = n_match / max(1, n_pred)
        msg = (f"ep {ep+1}/{args.epochs}  recall@0.5 {recall:.3f}  prec {precision:.3f}  "
               f"npred {n_pred}  ngt {n_gt}  time {(time.time()-t0)/60:.1f}min  lr {lr:.2e}")
        print(msg, flush=True); log_f.write(msg + "\n")

        torch.save({"epoch": ep, "model": model.state_dict()}, runs / "last.pt")
        if recall > best:
            best = recall
            torch.save({"epoch": ep, "model": model.state_dict(), "recall": recall},
                       runs / "best.pt")
            log_f.write(f"  -> new best recall {recall:.4f}\n")

    log_f.close()
    print(f"done. best recall@0.5 {best:.4f}")


def box_iou(a, b):
    """[P,4]×[G,4] → [P,G]."""
    a1 = a[:, :2]; a2 = a[:, 2:4]
    b1 = b[:, :2]; b2 = b[:, 2:4]
    inter = (torch.minimum(a2[:, None], b2[None]) -
             torch.maximum(a1[:, None], b1[None])).clamp(0).prod(-1)
    area_a = (a2 - a1).clamp(0).prod(-1)
    area_b = (b2 - b1).clamp(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-7)


def decode_batch(outs, score_th=0.5, nms_iou=0.4, max_per_img=200):
    """Decode multi-scale outputs to per-image bboxes+scores.

    Returns list of length B: each item = (boxes [N,4], scores [N]).
    """
    strides = (8, 16, 32)
    B = outs[0][0].shape[0]
    device = outs[0][0].device

    all_boxes = [[] for _ in range(B)]
    all_scores = [[] for _ in range(B)]

    for li, (c, b, k) in enumerate(outs):
        s = strides[li]
        H, W = c.shape[-2:]
        ys, xs = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing="ij")
        cx = (xs + 0.5).flatten() * s
        cy = (ys + 0.5).flatten() * s
        sc = torch.sigmoid(c).reshape(B, -1)
        bo = F.relu(b).permute(0, 2, 3, 1).reshape(B, -1, 4)
        for bi in range(B):
            keep = sc[bi] >= score_th
            if not keep.any(): continue
            cells = keep.nonzero(as_tuple=False).squeeze(1)
            l, t, r, bb = bo[bi, cells].unbind(-1)
            xb1 = cx[cells] - l * s; yb1 = cy[cells] - t * s
            xb2 = cx[cells] + r * s; yb2 = cy[cells] + bb * s
            boxes = torch.stack([xb1, yb1, xb2, yb2], -1)
            all_boxes[bi].append(boxes)
            all_scores[bi].append(sc[bi, cells])

    out = []
    for bi in range(B):
        if not all_boxes[bi]:
            out.append((torch.zeros(0, 4, device=device), torch.zeros(0, device=device)))
            continue
        boxes = torch.cat(all_boxes[bi], 0)
        scores = torch.cat(all_scores[bi], 0)
        keep = nms(boxes, scores, nms_iou, max_per_img)
        out.append((boxes[keep], scores[keep]))
    return out


def nms(boxes, scores, iou_th=0.4, max_k=200):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0].item()
        keep.append(i)
        if len(keep) >= max_k: break
        if order.numel() == 1: break
        a = boxes[i].unsqueeze(0)
        b = boxes[order[1:]]
        ious = box_iou(a, b).squeeze(0)
        order = order[1:][ious < iou_th]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


if __name__ == "__main__":
    main()
