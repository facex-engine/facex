"""
WFLW landmark dataset: mmap'd blob.jpg.bin + index.bin.

Index record: u64 offset | u32 size | f32[196] landmarks (in [0,1]).
"""

import io
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_PTS = 98
INDEX_REC = struct.Struct(f"<QI{NUM_PTS * 2}f")
INDEX_SIZE = INDEX_REC.size


class WFLWDataset(Dataset):
    def __init__(self, root: str, augment: bool = True, size: int = 112):
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.n = meta["n_faces"]
        self.size = size
        self.augment = augment
        self._blob_path = root / "blob.jpg.bin"
        self._idx_path = root / "index.bin"
        self._blob_mm = None
        self._idx_mm = None

    def _ensure_open(self):
        if self._blob_mm is None:
            self._blob_mm = np.memmap(self._blob_path, dtype=np.uint8, mode="r")
            self._idx_mm = np.memmap(self._idx_path, dtype=np.uint8, mode="r")

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        self._ensure_open()
        rec = self._idx_mm[i * INDEX_SIZE:(i + 1) * INDEX_SIZE].tobytes()
        unpacked = INDEX_REC.unpack(rec)
        offset, size = unpacked[0], unpacked[1]
        pts = np.array(unpacked[2:], dtype=np.float32).reshape(NUM_PTS, 2)

        jpeg = self._blob_mm[offset:offset + size].tobytes()
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)

        arr = np.asarray(img, dtype=np.uint8)  # HWC

        if self.augment:
            # tiny scale + translate jitter; we DON'T do flip here (would
            # need 98-point flip map). Light rotation by +/- 15deg.
            arr, pts = _augment(arr, pts)

        # normalize
        arr = (arr.astype(np.float32) - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))   # CHW
        return (torch.from_numpy(arr),
                torch.from_numpy(pts.reshape(-1).astype(np.float32)))


def _augment(img_hwc: np.ndarray, pts: np.ndarray):
    """Slight affine jitter applied identically to image and landmarks."""
    H, W = img_hwc.shape[:2]
    angle = (np.random.rand() - 0.5) * 30.0     # +/- 15 deg
    scale = 1.0 + (np.random.rand() - 0.5) * 0.2  # 0.9..1.1
    tx = (np.random.rand() - 0.5) * 0.08
    ty = (np.random.rand() - 0.5) * 0.08
    cx, cy = 0.5, 0.5
    cos = np.cos(np.deg2rad(angle)) * scale
    sin = np.sin(np.deg2rad(angle)) * scale
    M = np.array([[cos, -sin, cx + tx - cos * cx + sin * cy],
                  [sin,  cos, cy + ty - sin * cx - cos * cy]],
                 dtype=np.float32)
    # apply to landmarks (in normalized coords)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts2 = pts_h @ M.T

    # apply to image (in pixel coords). cv2 wants pixel-space affine.
    import cv2
    M_pix = M.copy()
    # rebuild in pixel coords (multiply translation by image size)
    M_pix[0, 2] *= W
    M_pix[1, 2] *= H
    img2 = cv2.warpAffine(img_hwc, M_pix, (W, H),
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return img2, pts2
