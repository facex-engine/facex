"""
Fast dataset: pre-decoded raw RGB blob + labels blob.
No JPEG decode per item — just a memmap slice and tensor convert.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_PTS = 478
LABEL_BYTES = NUM_PTS * 3 * 4
IMG_SIZE = 112
IMG_BYTES = IMG_SIZE * IMG_SIZE * 3

LABELS = Path("D:/apps/facex/training/data/mp_labels")


class MP3DDataset(Dataset):
    def __init__(self, augment: bool = True):
        meta = json.loads((LABELS / "meta.json").read_text())
        self.n = meta["n_records"]
        self.augment = augment
        self._raw_mm = None
        self._lbl_mm = None

    def _ensure(self):
        if self._raw_mm is None:
            self._raw_mm = np.memmap(LABELS / "raw.bin", dtype=np.uint8, mode="r",
                                      shape=(self.n, IMG_SIZE, IMG_SIZE, 3))
            self._lbl_mm = np.memmap(LABELS / "labels.bin", dtype=np.float32, mode="r",
                                      shape=(self.n, NUM_PTS, 3))

    def __len__(self): return self.n

    def __getitem__(self, i: int):
        self._ensure()
        # Copy out of the mmap so the tensor is owned + writable for augment.
        arr = np.array(self._raw_mm[i], dtype=np.uint8)         # [112, 112, 3]
        pts = np.array(self._lbl_mm[i], dtype=np.float32)        # [478, 3]

        if self.augment:
            arr, pts = _augment(arr, pts)

        arr = (arr.astype(np.float32) - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), torch.from_numpy(pts)


def _augment(img_hwc, pts):
    """Affine + perspective (synthetic yaw/pitch) augmentation. The synthetic
    yaw is the key addition over a frontal-only pipeline — it teaches the
    model to handle side-view faces that MS1M doesn't cover natively."""
    import cv2
    H, W = img_hwc.shape[:2]
    # ---- 1) affine: roll / scale / translate ----
    angle = (np.random.rand() - 0.5) * 16.0
    scale = 1.0 + (np.random.rand() - 0.5) * 0.15
    tx = (np.random.rand() - 0.5) * 0.05
    ty = (np.random.rand() - 0.5) * 0.05

    # ---- 2) synthetic yaw / pitch via perspective ----
    # Treat face as a 3D plane at z=0, rotate around y (yaw) and x (pitch),
    # project with simple perspective. Generates plausible side views.
    yaw   = (np.random.rand() - 0.5) * 60.0    # ±30°
    pitch = (np.random.rand() - 0.5) * 30.0    # ±15°

    # 4 source corners
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    cy_, cx_ = W / 2, W / 2
    f = W * 1.5
    cosy, siny = np.cos(np.radians(yaw)),   np.sin(np.radians(yaw))
    cosp, sinp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rx = np.array([[1, 0, 0], [0, cosp, -sinp], [0, sinp, cosp]])
    R3 = Ry @ Rx
    corners3d = np.array([[-cx_, -cy_, 0], [cx_, -cy_, 0],
                           [cx_,  cy_, 0], [-cx_,  cy_, 0]], dtype=np.float32)
    rot = corners3d @ R3.T
    z = rot[:, 2] + f
    x2 = rot[:, 0] * f / z + cx_
    y2 = rot[:, 1] * f / z + cy_
    persp_dst = np.float32(list(zip(x2, y2)))
    Hp = cv2.getPerspectiveTransform(src, persp_dst)

    # ---- 3) combine affine with perspective ----
    # affine in pixel coords
    aa = np.cos(np.deg2rad(angle)) * scale
    bb = np.sin(np.deg2rad(angle)) * scale
    Ma = np.array([
        [aa, -bb, (0.5 + tx - aa*0.5 + bb*0.5) * W],
        [bb,  aa, (0.5 + ty - bb*0.5 - aa*0.5) * H],
        [0,   0,  1.0],
    ], dtype=np.float32)
    M3 = Hp @ Ma     # apply affine first, then perspective

    img2 = cv2.warpPerspective(img_hwc, M3, (W, H),
                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # Transform landmarks: xy with full perspective M3 (in pixel space);
    # z scaled by avg horizontal compression (approximates depth foreshortening).
    xy_pix = pts[:, :2] * np.array([W, H], dtype=np.float32)
    ones = np.ones((xy_pix.shape[0], 1), dtype=np.float32)
    xy_h = np.concatenate([xy_pix, ones], axis=1)
    warped = xy_h @ M3.T
    xy_new = warped[:, :2] / np.maximum(warped[:, 2:3], 1e-6)
    pts2 = pts.copy()
    pts2[:, :2] = xy_new / np.array([W, H], dtype=np.float32)
    # adjust z by yaw magnitude (rough approximation — far side gets more depth)
    pts2[:, 2] = pts[:, 2] * (cosy * cosp)

    return img2, pts2
