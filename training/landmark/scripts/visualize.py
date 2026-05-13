"""Render predicted landmarks on a few test faces for sanity check."""
import io
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import LandmarkNet
from dataset import WFLWDataset

DATA = Path("D:/apps/facex/training/data/wflw/test")
CKPT = Path("D:/apps/facex/training/landmark/runs/best.pt")
OUT_DIR = Path("D:/apps/facex/training/landmark/runs/samples")
OUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LandmarkNet().to(device).eval()
ck = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ck["model"])
print(f"Loaded best.pt (NME {ck.get('nme', '?')*100:.2f}%)")

ds = WFLWDataset(DATA, augment=False)
import torch.utils.data as td
loader = td.DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
imgs, pts_true = next(iter(loader))
imgs = imgs.to(device)
with torch.no_grad():
    pred = model(imgs).cpu().numpy().reshape(-1, 98, 2)
pts_true = pts_true.numpy().reshape(-1, 98, 2)

# Denormalize and draw
imgs_np = ((imgs.cpu().numpy() * 128.0 + 127.5).clip(0, 255).astype(np.uint8))
for i in range(8):
    rgb = np.transpose(imgs_np[i], (1, 2, 0))
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for x, y in pts_true[i] * 112:
        d.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(0, 255, 0))
    for x, y in pred[i] * 112:
        d.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 80, 80))
    img.save(OUT_DIR / f"sample_{i}.png")
print(f"Wrote 8 samples to {OUT_DIR}")
print("Green = ground truth, Red = predicted")
