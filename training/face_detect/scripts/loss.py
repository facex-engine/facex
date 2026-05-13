"""
Anchor-free FCOS-style loss for face detection.

For each ground-truth box, we assign anchor cells that:
  - lie inside the box, AND
  - belong to the FPN level matching the box size:
      sqrt(area) <  64 → stride 8
      64 <= sqrt(area) < 128 → stride 16
      sqrt(area) >= 128 → stride 32

Each positive cell predicts:
  cls: 1-dim logit (face vs bg) → BCE focal loss
  box: (l, t, r, b) distances from cell center to box edges, in stride units → GIoU loss
  kps: (kx, ky) for each of 5 keypoints, normalized offsets from cell center
       in stride units → smooth L1 loss (only when keypoints are annotated)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ltrb_to_xyxy(centers, ltrb, stride):
    """Convert FCOS-style (l,t,r,b) distances back to absolute boxes (xyxy)."""
    cx = centers[..., 0]; cy = centers[..., 1]
    l, t, r, b = ltrb.unbind(-1)
    return torch.stack([cx - l * stride, cy - t * stride,
                         cx + r * stride, cy + b * stride], -1)


def _giou_loss(pred_xyxy, gt_xyxy, eps=1e-7):
    """Generalized IoU loss between predicted and gt boxes."""
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)
    ix1 = torch.maximum(px1, gx1); iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2); iy2 = torch.minimum(py2, gy2)
    iw = (ix2 - ix1).clamp(0); ih = (iy2 - iy1).clamp(0)
    inter = iw * ih
    parea = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    garea = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = parea + garea - inter + eps
    iou = inter / union
    cx1 = torch.minimum(px1, gx1); cy1 = torch.minimum(py1, gy1)
    cx2 = torch.maximum(px2, gx2); cy2 = torch.maximum(py2, gy2)
    carea = (cx2 - cx1) * (cy2 - cy1) + eps
    giou = iou - (carea - union) / carea
    return 1.0 - giou


def _focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    """Focal binary cross-entropy. logits and targets are flat tensors."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    return (w * (1 - pt).pow(gamma) * ce).sum()


def assign_level(box):
    """Return stride index (0,1,2) for a box [x1,y1,x2,y2]."""
    w = box[2] - box[0]; h = box[3] - box[1]
    s = (w * h).clamp(min=1).sqrt()
    if s < 64: return 0
    if s < 128: return 1
    return 2


class DetectorLoss(nn.Module):
    """Computes loss given list of model outputs and per-image GT lists.

    Outputs from FaceDetector: tuple of 3 tuples, each (cls, box, kps) of shape
        cls: [B, 1, H, W]
        box: [B, 4, H, W]
        kps: [B, 10, H, W]
    """
    STRIDES = (8, 16, 32)

    def __init__(self, input_size=320):
        super().__init__()
        self.input_size = input_size
        # Precompute anchor centers per level.
        self._anchors = {}
        for s in self.STRIDES:
            n = input_size // s
            ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
            c = (torch.stack([xs.flatten(), ys.flatten()], -1).float() + 0.5) * s
            self._anchors[s] = c  # [H*W, 2]

    def _anchors_on(self, device):
        return {s: a.to(device) for s, a in self._anchors.items()}

    def forward(self, outs, gt_boxes, gt_kps):
        """outs: list of 3 (cls, box, kps).  gt_boxes/kps: list of B tensors."""
        device = outs[0][0].device
        B = outs[0][0].shape[0]
        anchors = self._anchors_on(device)

        # Flatten predictions across spatial dims
        # Per level: cls [B, H*W], box [B, H*W, 4], kps [B, H*W, 10]
        flat = []
        for li, (c, b, k) in enumerate(outs):
            s = self.STRIDES[li]
            H, W = c.shape[-2:]
            flat.append({
                "stride": s,
                "cls": c.permute(0, 2, 3, 1).reshape(B, -1),
                "box": b.permute(0, 2, 3, 1).reshape(B, -1, 4),
                "kps": k.permute(0, 2, 3, 1).reshape(B, -1, 10),
                "centers": anchors[s],   # [H*W, 2]
            })

        cls_loss_total = torch.tensor(0.0, device=device)
        box_loss_total = torch.tensor(0.0, device=device)
        kps_loss_total = torch.tensor(0.0, device=device)
        n_pos_total = 0
        n_kps_total = 0

        # Build per-image targets level-by-level
        for bi in range(B):
            gtb = gt_boxes[bi].to(device)
            gtk = gt_kps[bi].to(device)
            if gtb.shape[0] == 0:
                # No faces: all negatives
                for L in flat:
                    cls_loss_total = cls_loss_total + _focal_bce(
                        L["cls"][bi], torch.zeros_like(L["cls"][bi]))
                continue
            # Assign each GT to a level by size
            level_of = torch.tensor([assign_level(b) for b in gtb], device=device)
            for li, L in enumerate(flat):
                s = L["stride"]
                centers = L["centers"]                    # [N, 2]
                cls_pred = L["cls"][bi]                   # [N]
                box_pred = L["box"][bi]                   # [N, 4]
                kps_pred = L["kps"][bi]                   # [N, 10]

                # Targets for this level: only GTs assigned to it
                this_mask = (level_of == li)
                if not this_mask.any():
                    cls_loss_total = cls_loss_total + _focal_bce(
                        cls_pred, torch.zeros_like(cls_pred))
                    continue

                gt_this = gtb[this_mask]
                kp_this = gtk[this_mask]   # [G, 5, 2]

                # For each anchor, find the GT (assigned to this level) it belongs to.
                # Positive if anchor center is inside any GT box.
                # When multiple GTs contain a cell, pick the smaller box (FCOS rule).
                cx = centers[:, 0:1]    # [N,1]
                cy = centers[:, 1:2]
                gx1 = gt_this[:, 0]; gy1 = gt_this[:, 1]
                gx2 = gt_this[:, 2]; gy2 = gt_this[:, 3]
                # Distances inside box
                l = cx - gx1; t = cy - gy1; r = gx2 - cx; b = gy2 - cy
                inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)  # [N, G]
                # GT area for tiebreak
                gt_area = (gx2 - gx1) * (gy2 - gy1)
                # Mask: select smallest box that contains the cell
                area_mat = inside.float() * (1.0 / (gt_area + 1e-6))  # higher = smaller box
                best_gt = area_mat.argmax(1)                          # [N]
                pos_mask = inside.any(dim=1)                          # [N]

                # cls target
                cls_target = pos_mask.float()
                cls_loss_total = cls_loss_total + _focal_bce(cls_pred, cls_target)

                if pos_mask.any():
                    pi = pos_mask.nonzero(as_tuple=False).squeeze(1)
                    gi = best_gt[pi]
                    # GT box → distances in stride units
                    gb = gt_this[gi]
                    cl = centers[pi]
                    ltrb_gt = torch.stack([
                        (cl[:, 0] - gb[:, 0]) / s,
                        (cl[:, 1] - gb[:, 1]) / s,
                        (gb[:, 2] - cl[:, 0]) / s,
                        (gb[:, 3] - cl[:, 1]) / s,
                    ], -1)
                    # Predicted distances (model outputs ARE in stride units, after relu)
                    ltrb_pr = F.relu(box_pred[pi])

                    pred_xyxy = _ltrb_to_xyxy(cl.unsqueeze(0), ltrb_pr.unsqueeze(0), s).squeeze(0)
                    box_loss_total = box_loss_total + _giou_loss(pred_xyxy, gb).sum()
                    n_pos_total += pi.numel()

                    # Landmarks (only for GTs with valid keypoints)
                    kp_gt = kp_this[gi]            # [P, 5, 2]
                    has_kps = ~torch.isnan(kp_gt[:, 0, 0])
                    if has_kps.any():
                        idx = has_kps.nonzero(as_tuple=False).squeeze(1)
                        kp_target_xy = kp_gt[idx]    # [Q, 5, 2]
                        # Encode as offsets from cell center, in stride units
                        cl_q = cl[idx].unsqueeze(1)  # [Q, 1, 2]
                        off_gt = (kp_target_xy - cl_q) / s                # [Q, 5, 2]
                        kps_pr = kps_pred[pi[idx]].view(-1, 5, 2)
                        kps_loss_total = kps_loss_total + F.smooth_l1_loss(
                            kps_pr, off_gt, reduction="sum")
                        n_kps_total += idx.numel() * 5

        # Normalize
        n_pos = max(1, n_pos_total)
        cls_loss = cls_loss_total / n_pos
        box_loss = box_loss_total / n_pos
        kps_loss = kps_loss_total / max(1, n_kps_total)

        total = cls_loss + 1.5 * box_loss + 0.5 * kps_loss
        return total, {"cls": cls_loss.item(), "box": box_loss.item(),
                       "kps": kps_loss.item(), "n_pos": n_pos_total}
