"""Memory-mapped antispoof dataset loader."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path("D:/apps/facex/training/data/antispoof")
SIZE = 112
IMG_BYTES = SIZE * SIZE * 3


class AntiSpoofDataset(Dataset):
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
            # color jitter
            if np.random.rand() < 0.8:
                arr = arr.astype(np.float32)
                arr *= np.random.uniform(0.85, 1.15)
                arr += np.random.uniform(-15, 15, size=3)[None, None]
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        arr = (arr.astype(np.float32) - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)
