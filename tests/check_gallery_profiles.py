"""Exercise the production and FP32 gallery paths on the same input bytes."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from bench_gallery_groups import embed_patches, fixture_patches, prepare


def run_engine(weights, patches, executable):
    views = embed_patches(weights, patches, executable.resolve())
    if not np.isfinite(views).all():
        raise ValueError(f"{executable} produced a nonfinite embedding")
    print(f"{executable.name} embeddings_sha256={hashlib.sha256(views.tobytes()).hexdigest()}", flush=True)
    return views


def directions(views):
    vectors = views.reshape(-1, 512).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero fixture embedding")
    return vectors / norms


def check_fp32(reference, native):
    left, right = directions(reference), directions(native)
    coordinate_drift = np.max(np.abs(left - right))
    cosine_drift = np.max(np.abs(left @ left.T - right @ right.T))
    print(f"FP32 max coordinate drift={coordinate_drift:.8g} max pairwise cosine drift={cosine_drift:.8g}", flush=True)
    if coordinate_drift > 1e-5 or cosine_drift > 1e-5:
        raise AssertionError("FP32 build profiles diverged")


def write_gallery(path, views):
    path.mkdir(parents=True, exist_ok=True)
    prepare(path, views, scaled=True)
    near_and_boundary = np.load(path / "queries.npy")
    random = np.random.default_rng(77).standard_normal((15, 512))
    np.vstack((near_and_boundary, random)).astype("<f4").tofile(path / "queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("weights", "production", "reference", "native", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    patches = fixture_patches()
    print(f"crops_sha256={hashlib.sha256(patches).hexdigest()}", flush=True)
    reference = run_engine(args.weights, patches, args.reference)
    native = run_engine(args.weights, patches, args.native)
    check_fp32(reference, native)
    write_gallery(args.output / "fp32", native)
    write_gallery(args.output / "production", run_engine(args.weights, patches, args.production))
