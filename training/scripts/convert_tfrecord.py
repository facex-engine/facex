"""
Convert MS1M-RefineV2 tfrecord -> fast on-disk format for training.

Output layout under D:/apps/facex/training/data/ms1m/:
    blob.jpg.bin    concatenated raw jpeg bytes (no decoding, no rewrite)
    index.bin       struct-packed records: (offset:u64, size:u32, label:u32)
    meta.json       {"n_images": N, "n_classes": K, "labels_min/max": ...}

Usage:
    python convert_tfrecord.py D:/archive.zip
or, if the .tfrecords is already extracted:
    python convert_tfrecord.py D:/path/to/ms1m_refined_v2_bin.tfrecords
"""

import io
import json
import os
import struct
import sys
import time
import zipfile
from pathlib import Path

import tensorflow as tf  # only for tf.train.Example parsing


OUT_DIR = Path("D:/apps/facex/training/data/ms1m")


def open_tfrecord_source(path: str):
    """Return a tf.data.TFRecordDataset, handling .zip wrappers transparently."""
    if path.lower().endswith(".zip"):
        # Extract the .tfrecords once next to the zip if not already done
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".tfrecords")]
            if not names:
                raise SystemExit(f"No .tfrecords in {path}")
            extracted = Path(path).parent / names[0]
            if not extracted.exists():
                print(f"Extracting {names[0]} from zip ({z.getinfo(names[0]).file_size/1e9:.1f} GB)...")
                z.extract(names[0], path=Path(path).parent)
            return str(extracted)
    return path


def parse_example(raw_record: bytes):
    """Pull (jpeg_bytes, label:int) out of a tf.train.Example."""
    ex = tf.train.Example()
    ex.ParseFromString(raw_record)
    feats = ex.features.feature
    # InsightFace MS1M-V2 tfrecord typically uses 'image_raw' + 'label'
    # but some variants use 'image' / 'class_id'. Detect once.
    img = (feats.get("image_raw") or feats.get("image"))
    lbl = (feats.get("label") or feats.get("class_id"))
    if img is None or lbl is None:
        raise SystemExit(f"unknown schema, keys: {list(feats.keys())}")
    jpeg = img.bytes_list.value[0]
    label = int(lbl.int64_list.value[0])
    return jpeg, label


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = open_tfrecord_source(sys.argv[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blob_path = OUT_DIR / "blob.jpg.bin"
    idx_path = OUT_DIR / "index.bin"
    meta_path = OUT_DIR / "meta.json"

    if blob_path.exists() and idx_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Already converted: {meta}")
        return

    print(f"Reading: {src}")
    ds = tf.data.TFRecordDataset(src, buffer_size=64 * 1024 * 1024)

    n = 0
    labels_seen = set()
    label_min, label_max = 10**12, -1
    t0 = time.time()
    offset = 0
    blob_f = open(blob_path, "wb", buffering=8 * 1024 * 1024)
    idx_f = open(idx_path, "wb", buffering=4 * 1024 * 1024)
    pack = struct.Struct("<QII").pack  # offset, size, label

    for raw in ds:
        jpeg, label = parse_example(raw.numpy())
        sz = len(jpeg)
        blob_f.write(jpeg)
        idx_f.write(pack(offset, sz, label))
        offset += sz
        n += 1
        if label > label_max: label_max = label
        if label < label_min: label_min = label
        labels_seen.add(label)
        if n % 50000 == 0:
            dt = time.time() - t0
            mb = offset / 1e6
            print(f"  {n:>8,} imgs | {mb:>7.1f} MB | "
                  f"{n/dt:>6.0f} img/s | classes seen: {len(labels_seen):,}")

    blob_f.close()
    idx_f.close()
    meta = {
        "n_images": n,
        "n_classes": len(labels_seen),
        "label_min": label_min,
        "label_max": label_max,
        "blob_bytes": offset,
        "blob_path": str(blob_path),
        "index_path": str(idx_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Done. {n:,} images, {len(labels_seen):,} classes, blob {offset/1e9:.2f} GB")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
