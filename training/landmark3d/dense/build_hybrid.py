"""
Hybrid template: 478 MP vertices warped by 98 WFLW controls via TPS.

Idea: WFLW model is pose-robust (trained on WFLW which has poses), MP-distilled
model only has frontal density. We use the MP CANONICAL template (478 vertices
in average-face space) and per-frame warp it via TPS using the WFLW model's
output as control points.

Result: at runtime, only the 98-pt WFLW model needs to run. Dense 478 mesh
emerges from TPS deformation of the canonical MP template — automatically
follows head pose because WFLW does.

Offline steps:
  1. Compute canonical MP positions = mean of all 60K MP labels (frontal MS1M).
  2. Compute canonical WFLW positions = mean of WFLW training set
     (already saved as canonical_98.npy).
  3. Map each WFLW index to its nearest MP vertex index (semi-anatomical).
  4. Compute TPS transfer T (478 × 98) from canonical (X_wflw, X_mp_canonical).

Output:
  D:/apps/facex/wasm/wflw_to_mp_hybrid.json
"""

import json
import struct
from pathlib import Path

import numpy as np


LABELS = Path("D:/apps/facex/training/data/mp_labels")
CANON_WFLW = Path("D:/apps/facex/training/landmark/dense/canonical_98.npy")
OUT = Path("D:/apps/facex/wasm/wflw_to_mp_hybrid.json")

N_MP = 478
LBL_BYTES = N_MP * 3 * 4


def main():
    # ----- canonical MP positions (mean over training labels) -----
    n = (LABELS / "labels.bin").stat().st_size // LBL_BYTES
    lbl_mm = np.memmap(LABELS / "labels.bin", dtype=np.float32, mode="r",
                        shape=(n, N_MP, 3))
    canon_mp = lbl_mm.mean(axis=0).astype(np.float32)         # [478, 3]
    print(f"canonical MP: {canon_mp.shape}, x=[{canon_mp[:,0].min():.2f},{canon_mp[:,0].max():.2f}], "
          f"y=[{canon_mp[:,1].min():.2f},{canon_mp[:,1].max():.2f}], z=[{canon_mp[:,2].min():.2f},{canon_mp[:,2].max():.2f}]")

    # ----- canonical WFLW positions (already saved as mean) -----
    canon_wflw = np.load(CANON_WFLW).astype(np.float32)         # [98, 2]

    # Rescale canonical MP so its KEY anatomical anchors align with WFLW's:
    # - WFLW eye center  (avg of pupils 96, 97)        <-> MP avg of (468, 473)
    # - WFLW mouth center (avg of mouth corners 76,82) <-> MP avg of (61, 291)
    # - inter-pupil distance for X-scale
    # Forehead, ears, chin then extend BEYOND WFLW bbox naturally
    # (because MP has those landmarks, WFLW doesn't).
    wflw_eye_c   = (canon_wflw[96]  + canon_wflw[97]) * 0.5
    wflw_mouth_c = (canon_wflw[76]  + canon_wflw[82]) * 0.5
    wflw_eye_dx  = abs(canon_wflw[97][0] - canon_wflw[96][0])

    mp_eye_c   = (canon_mp[468, :2] + canon_mp[473, :2]) * 0.5
    mp_mouth_c = (canon_mp[61,  :2] + canon_mp[291, :2]) * 0.5
    mp_eye_dx  = abs(canon_mp[473, 0] - canon_mp[468, 0])

    # Scale X by inter-pupil distance, Y by eye->mouth distance
    sx = wflw_eye_dx / max(mp_eye_dx, 1e-6)
    sy = (wflw_mouth_c[1] - wflw_eye_c[1]) / max(mp_mouth_c[1] - mp_eye_c[1], 1e-6)

    # Translate so eye centers coincide
    tx = wflw_eye_c[0] - mp_eye_c[0] * sx
    ty = wflw_eye_c[1] - mp_eye_c[1] * sy

    canon_mp[:, 0] = canon_mp[:, 0] * sx + tx
    canon_mp[:, 1] = canon_mp[:, 1] * sy + ty
    canon_mp[:, 2] = canon_mp[:, 2] * ((sx + sy) * 0.5)
    print(f"rescaled MP anchored on eyes+mouth: sx={sx:.3f} sy={sy:.3f}")
    print(f"after rescale: MP x=[{canon_mp[:,0].min():.2f},{canon_mp[:,0].max():.2f}], "
          f"y=[{canon_mp[:,1].min():.2f},{canon_mp[:,1].max():.2f}] (forehead extends above)")

    # ----- Hand-crafted WFLW -> MP mapping (strict anatomical correspondence) -----
    # MediaPipe vertex indices for each WFLW point. Standard MP vertex semantics
    # come from the official face mesh diagram. Where exact mapping is unclear,
    # we pick the closest semantic neighbor.
    canon_mp_xy = canon_mp[:, :2]
    WFLW_TO_MP = [
        # ----- Face contour 0..32 (left ear -> chin -> right ear) -----
        # MP face_oval order: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        #                     361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        #                     176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        #                     162, 21, 54, 103, 67, 109
        # WFLW contour starts at left ear top (idx 0) and goes down jaw to chin
        # (idx 16) and back up right side to right ear top (idx 32).
        127, 234, 93, 132, 58, 172, 136, 150,    # 0..7  left ear -> upper jaw
        149, 176, 148, 152, 377, 400, 378, 379,  # 8..15 lower jaw -> chin region
        152, 377, 378, 400, 379, 365, 397, 288,  # 16..23 chin -> right jaw
        361, 323, 454, 356, 389, 251, 284, 332,  # 24..31 right jaw -> right ear
        297,                                       # 32 right ear top
        # ----- Left eyebrow 33..41 (outer -> inner -> back along bottom) -----
        70, 63, 105, 66, 107,    # 33..37 upper L brow (outer -> inner)
        55, 65, 52, 53,          # 38..41 lower L brow (inner -> outer back)
        # ----- Right eyebrow 42..50 -----
        285, 295, 282, 283, 276, # 42..46 (inner -> outer along upper)
        300, 293, 334, 296,      # 47..50 lower R brow
        # ----- Nose bridge 51..54 (top -> tip) -----
        168, 197, 195, 5,
        # ----- Nose bottom 55..59 (L nostril -> tip -> R nostril) -----
        240, 219, 1, 459, 460,
        # ----- Left eye 60..67 (clockwise from outer corner) -----
        33, 160, 158, 133, 153, 144, 145, 163,
        # ----- Right eye 68..75 -----
        362, 385, 387, 263, 373, 374, 381, 390,
        # ----- Outer mouth 76..87 (clockwise from L corner) -----
        61, 39, 37, 0, 267, 269, 270, 291, 314, 17, 84, 181,
        # ----- Inner mouth 88..95 -----
        78, 82, 13, 312, 308, 317, 14, 87,
        # ----- Pupils 96, 97 -----
        468, 473,
    ]
    assert len(WFLW_TO_MP) == 98, f"mapping length {len(WFLW_TO_MP)} != 98"
    mapping = np.array(WFLW_TO_MP, dtype=np.int32)
    unique = len(set(WFLW_TO_MP))
    print(f"WFLW->MP mapping: {unique} unique MP vertices (collisions: {98 - unique})")

    # ----- TPS transfer: from canonical_98 (control) to all 478 (target) -----
    # canonical_98_at_mp = the MP-canonical positions of the 98 mapped vertices.
    # We learn TPS such that for each mp_idx, dense[mp_idx] = f(canonical_wflw[mp_idx])
    # where f interpolates from (canonical_wflw -> canonical_mp_xy).
    # In practice: given new WFLW positions, compute MP positions via TPS.

    N = 98
    Xc = canon_wflw                                              # [98, 2]
    Yc_xy = canon_mp_xy[mapping]                                  # [98, 2]  — MP-canonical positions of mapped indices
    Yc_z  = canon_mp[mapping, 2:3]                                # [98, 1]

    def phi(r):
        return np.where(r > 1e-12, r * r * np.log(r + 1e-12), 0.0)

    K = phi(np.linalg.norm(Xc[:, None] - Xc[None], axis=2))
    P = np.hstack([np.ones((N, 1)), Xc])
    L = np.vstack([np.hstack([K, P]), np.hstack([P.T, np.zeros((3, 3))])])
    L_reg = L + 1e-7 * np.eye(N + 3)
    L_inv_left = np.linalg.solve(L_reg, np.vstack([np.eye(N), np.zeros((3, N))]))

    # For each MP target point j, compute its row [phi(p_mp_j - x_wflw_i), 1, p_mp_j_x, p_mp_j_y]
    # giving the contribution from each WFLW control point.
    # Here the "target" of each row is canon_mp_xy[j], so this builds the WARP from
    # canonical_wflw (control source) to canonical_mp (target).
    diffs = canon_mp_xy[:, None] - Xc[None]
    K_pj = phi(np.linalg.norm(diffs, axis=2))
    P_pj = np.hstack([np.ones((N_MP, 1)), canon_mp_xy])
    T = np.hstack([K_pj, P_pj]) @ L_inv_left                      # [478, 98]
    # Sanity: T @ canon_wflw ≈ canon_mp_xy
    err = np.max(np.abs(T @ Xc - canon_mp_xy))
    print(f"TPS recon err: {err:.6e}")

    # Triangulation: use MP TESSELATION (already in facemesh_topology.json)
    # — we don't need our own triangulation here, just the transfer matrix.

    # Per-vertex canonical Z (constant for rendering shading)
    z_canon = canon_mp[:, 2].astype(np.float32)
    # Normalize Z to [0, 1] (forward = 1)
    z_min, z_max = z_canon.min(), z_canon.max()
    z_norm = 1 - (z_canon - z_min) / max(z_max - z_min, 1e-6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "version": "wflw_hybrid_v1",
        "n_mp": int(N_MP),
        "n_wflw": int(N),
        "canonical_mp": canon_mp.tolist(),
        "canonical_wflw": canon_wflw.tolist(),
        "T": T.astype(np.float32).reshape(-1).tolist(),
        "z_norm": z_norm.tolist(),
    }, open(OUT, "w"))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
