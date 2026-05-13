"""
Memory-mapped dataset reader for the converted MS1M blob.

Provides a torch IterableDataset / Dataset that decodes JPEG on the fly
in worker processes. Designed for fast random access without keeping
the whole 16 GB in RAM.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import io

# 9-byte struct: 8 (offset u64) + 4 (size u32) + 4 (label u32) = 16 bytes
INDEX_REC = struct.Struct("<QII")
INDEX_SZ = INDEX_REC.size  # 16


class MS1MDataset(Dataset):
    def __init__(self, root: str, augment: bool = True, size: int = 112):
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.n = meta["n_images"]
        self.n_classes = meta["n_classes"]
        self.size = size
        self.augment = augment
        # Memory-map the raw bytes — OS handles paging
        self._blob_path = root / "blob.jpg.bin"
        self._idx_path = root / "index.bin"
        self._blob_mm = None
        self._idx_mm = None

    def _ensure_open(self):
        # Open in worker, not __init__ (mmap not picklable across fork on Win)
        if self._blob_mm is None:
            self._blob_mm = np.memmap(self._blob_path, dtype=np.uint8, mode="r")
            self._idx_mm = np.memmap(self._idx_path, dtype=np.uint8, mode="r")

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        self._ensure_open()
        rec_off = i * INDEX_SZ
        offset, size, label = INDEX_REC.unpack(self._idx_mm[rec_off:rec_off + INDEX_SZ].tobytes())
        jpeg = self._blob_mm[offset:offset + size].tobytes()
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        # Light augmentation: random horizontal flip (the standard for face recog)
        if self.augment and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img, dtype=np.uint8)  # HWC, 0..255
        # CHW float32 normalized to [-1, 1]  (InsightFace convention)
        arr = arr.astype(np.float32)
        arr = (arr - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr), int(label)
