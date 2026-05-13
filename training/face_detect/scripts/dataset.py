"""
WIDER FACE dataset with RetinaFace-style 5-point landmarks.

Annotation format (label.txt from biubug6/Pytorch_Retinaface):
  # path/to/image.jpg
  x y w h  kp1x kp1y v1  kp2x kp2y v2  kp3x kp3y v3  kp4x kp4y v4  kp5x kp5y v5  blur
  x y w h  ...

kp visibility flag v: 0 = labeled, 1 = unlabeled. If any kp is unlabeled
(typical for tiny/occluded faces), we mark landmarks as missing (-1).

Output per item:
  image:  uint8 HxWx3 RGB → normalized float (B,3,H,W)
  bboxes: (N,4) in [x1, y1, x2, y2] pixel coords
  kps:    (N,5,2) pixel coords, NaN if not annotated
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random


ROOT = Path("D:/apps/facex/training/data/widerface")


def _parse_retinaface_label(path: Path):
    """Return list of (relative_image_path, faces). faces is list of dicts."""
    items = []
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                if current is not None: items.append(current)
                current = (line[1:].strip(), [])
                continue
            parts = line.split()
            nums = list(map(float, parts))
            x, y, w, h = nums[:4]
            face = {"box": [x, y, x + w, y + h]}
            if len(nums) >= 19:
                kps = np.array(nums[4:19], dtype=np.float32).reshape(5, 3)
                if (kps[:, 2] >= 0).all():
                    face["kps"] = kps[:, :2]
                else:
                    face["kps"] = None
            else:
                face["kps"] = None
            current[1].append(face)
    if current is not None: items.append(current)
    return items


def _parse_wider_official(path: Path):
    """Parse WIDER FACE official wider_face_*_bbx_gt.txt format.

    Format:
        <image_path>
        <num_faces>
        x y w h blur expr illu invalid occ pose   (one line per face; many flags)
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        rel = lines[i]; i += 1
        n = int(lines[i]); i += 1
        faces = []
        if n == 0:
            # Bug in WIDER: file lists 0 then has one zero-line; skip it
            if i < len(lines) and lines[i].startswith("0 0 0 0"):
                i += 1
        for _ in range(n):
            parts = list(map(int, lines[i].split())); i += 1
            x, y, w, h = parts[:4]
            invalid = parts[7] if len(parts) > 7 else 0
            if w <= 0 or h <= 0 or invalid: continue
            faces.append({"box": [x, y, x + w, y + h], "kps": None})
        items.append((rel, faces))
    return items


class WiderFaceDataset(Dataset):
    """Reads label.txt and pairs with WIDER_train or WIDER_val image directories."""
    def __init__(self, split: str = "train", input_size: int = 320, augment: bool = True):
        self.input_size = input_size
        self.augment = augment
        # Prefer RetinaFace 5-kps annotations if present, else fall back to
        # official WIDER FACE bbox-only annotations.
        rf_train = ROOT / "retinaface_gt_v1.1" / "train" / "label.txt"
        rf_val = ROOT / "retinaface_gt_v1.1" / "val" / "label.txt"
        of_train = ROOT / "wider_face_split" / "wider_face_train_bbx_gt.txt"
        of_val = ROOT / "wider_face_split" / "wider_face_val_bbx_gt.txt"

        if split == "train":
            self.img_root = ROOT / "WIDER_train" / "images"
            if rf_train.exists():
                self.items = _parse_retinaface_label(rf_train)
            elif of_train.exists():
                self.items = _parse_wider_official(of_train)
            else:
                raise FileNotFoundError(f"no train labels found in {ROOT}")
        else:
            self.img_root = ROOT / "WIDER_val" / "images"
            if rf_val.exists():
                self.items = _parse_retinaface_label(rf_val)
            elif of_val.exists():
                self.items = _parse_wider_official(of_val)
            else:
                raise FileNotFoundError(f"no val labels found in {ROOT}")
        # Drop entries with no faces or images we can't find
        self.items = [
            (p, faces) for p, faces in self.items
            if faces and (self.img_root / p).exists()
        ]
        print(f"[{split}] dataset: {len(self.items)} images")

    def __len__(self): return len(self.items)

    def _augment(self, img: np.ndarray, boxes: np.ndarray, kps: np.ndarray):
        """Random crop with at least one face, horizontal flip, photometric jitter."""
        H, W = img.shape[:2]
        N = len(boxes)
        if N == 0:
            return img, boxes, kps

        # Random crop: try sizes 0.3..1.0 of min(H,W), keep at least one face fully inside.
        for _ in range(15):
            scale = random.uniform(0.4, 1.0)
            side = int(min(H, W) * scale)
            cx = random.randint(side // 2, W - side // 2)
            cy = random.randint(side // 2, H - side // 2)
            x1, y1 = cx - side // 2, cy - side // 2
            x2, y2 = x1 + side, y1 + side
            # Keep faces fully inside crop
            keep_mask = (boxes[:, 0] >= x1) & (boxes[:, 1] >= y1) & \
                        (boxes[:, 2] <= x2) & (boxes[:, 3] <= y2)
            if keep_mask.any():
                img = img[y1:y2, x1:x2]
                boxes = boxes[keep_mask].copy()
                boxes[:, [0, 2]] -= x1; boxes[:, [1, 3]] -= y1
                kps2 = kps[keep_mask].copy()
                valid = ~np.isnan(kps2[..., 0])
                kps2[valid, 0] -= x1; kps2[valid, 1] -= y1
                kps = kps2
                break

        # Horizontal flip
        if random.random() < 0.5:
            h_, w_ = img.shape[:2]
            img = img[:, ::-1].copy()
            boxes[:, [0, 2]] = w_ - boxes[:, [2, 0]]
            valid = ~np.isnan(kps[..., 0])
            kps[valid, 0] = w_ - kps[valid, 0]
            # Swap left/right keypoints: 0↔1 (eyes), 3↔4 (mouth corners)
            kps = kps[:, [1, 0, 2, 4, 3]]

        # Photometric jitter (brightness/contrast/saturation on uint8)
        if random.random() < 0.6:
            arr = img.astype(np.float32)
            arr = arr * random.uniform(0.7, 1.3) + random.uniform(-20, 20)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = arr

        return img, boxes, kps

    def __getitem__(self, idx):
        rel, faces = self.items[idx]
        img = np.asarray(Image.open(self.img_root / rel).convert("RGB"))
        H0, W0 = img.shape[:2]

        boxes = np.array([f["box"] for f in faces], dtype=np.float32)   # N,4
        kps = np.full((len(faces), 5, 2), np.nan, dtype=np.float32)
        for i, f in enumerate(faces):
            if f["kps"] is not None: kps[i] = f["kps"]

        # Filter tiny boxes (likely "ignore" in WIDER FACE — area < 6×6 pixels)
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        big = (wh[:, 0] > 5) & (wh[:, 1] > 5)
        boxes = boxes[big]; kps = kps[big]

        if self.augment:
            img, boxes, kps = self._augment(img, boxes, kps)

        # Resize to input_size with letterbox preserving aspect ratio
        H, W = img.shape[:2]
        if H != self.input_size or W != self.input_size:
            scale = self.input_size / max(H, W)
            new_h, new_w = int(H * scale), int(W * scale)
            img_pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
            # pad to square with constant 0
            canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            canvas[:new_h, :new_w] = np.asarray(img_pil)
            img = canvas
            boxes = boxes * scale
            valid = ~np.isnan(kps[..., 0])
            kps[valid] *= scale

        # Clip boxes to image
        if len(boxes):
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.input_size - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.input_size - 1)
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            valid = (wh[:, 0] > 4) & (wh[:, 1] > 4)
            boxes = boxes[valid]; kps = kps[valid]

        # Normalize to [-1, 1]
        x = (img.astype(np.float32) - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))   # CHW

        return torch.from_numpy(x), torch.from_numpy(boxes), torch.from_numpy(kps)


def collate(batch):
    """Variable face count per image — keep as list of tensors."""
    imgs, boxes, kps = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(boxes), list(kps)


if __name__ == "__main__":
    ds = WiderFaceDataset("train", augment=True)
    print(f"sample: {len(ds)} items")
    img, b, k = ds[0]
    print(f"img: {img.shape}, boxes: {b.shape}, kps: {k.shape}")
    print(f"sample box: {b[0] if len(b) else 'none'}")
