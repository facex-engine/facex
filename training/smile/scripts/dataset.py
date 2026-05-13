"""Memory-mapped tongue dataset loader with strong augmentation."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

ROOT = Path("D:/apps/facex/training/data/smile")
SIZE = 112


class SmileDataset(Dataset):
    def __init__(self, augment: bool = True):
        meta = json.loads((ROOT / "meta.json").read_text())
        self.n = meta["n_samples"]
        self.augment = augment
        self._raw = None
        self._lbl = None

    def _ensure(self):
        if self._raw is None:
            self._raw = np.memmap(ROOT / "train_raw.bin", dtype=np.uint8, mode="r",
                                   shape=(self.n, SIZE, SIZE, 3))
            self._lbl = np.memmap(ROOT / "train_labels.bin", dtype=np.uint8, mode="r",
                                   shape=(self.n,))

    def __len__(self): return self.n

    def __getitem__(self, i):
        self._ensure()
        arr = np.array(self._raw[i], dtype=np.uint8)
        label = int(self._lbl[i])

        if self.augment:
            # horizontal flip
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1, :].copy()
            # random crop & resize (±15%)
            if np.random.rand() < 0.7:
                s = np.random.uniform(0.85, 1.0)
                side = int(SIZE * s)
                ox = np.random.randint(0, SIZE - side + 1)
                oy = np.random.randint(0, SIZE - side + 1)
                arr = cv2.resize(arr[oy:oy+side, ox:ox+side], (SIZE, SIZE))
            # color jitter
            if np.random.rand() < 0.85:
                arr = arr.astype(np.float32)
                arr *= np.random.uniform(0.75, 1.25)
                arr += np.random.uniform(-25, 25, size=3)[None, None]
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            # mild blur
            if np.random.rand() < 0.3:
                arr = cv2.GaussianBlur(arr, (3, 3), 0)
            # mild rotation
            if np.random.rand() < 0.4:
                angle = np.random.uniform(-12, 12)
                M = cv2.getRotationMatrix2D((SIZE/2, SIZE/2), angle, 1.0)
                arr = cv2.warpAffine(arr, M, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)

        x = (arr.astype(np.float32) - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)
