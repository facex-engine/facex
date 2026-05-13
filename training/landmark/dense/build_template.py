"""
Build the dense face mesh template.

1. Load canonical 98-point WFLW mean face (in [0,1] crop-relative coords).
2. Generate ~1000 virtual point positions across anatomical regions, all
   defined parametrically from the canonical 98 control points.
3. Build a fixed triangle topology connecting real + virtual points.
4. Precompute the TPS transfer matrix T such that, given the *current*
   frame's 98 WFLW points Y (shape [98, 2]), the dense mesh positions
   are simply  V = T @ Y  (shape [N_total, 2]).

Output: D:/apps/facex/wasm/face_mesh_template.json
"""

import json
from pathlib import Path

import numpy as np


CANON = Path("D:/apps/facex/training/landmark/dense/canonical_98.npy")
OUT = Path("D:/apps/facex/wasm/face_mesh_template.json")


# ============== generate virtual points (in canonical space) ==============

def catmull_rom(p0, p1, p2, p3, t):
    """Catmull-Rom interpolation at t in [0, 1] between p1 and p2."""
    t2 = t * t; t3 = t2 * t
    a = -0.5 * t3 + t2 - 0.5 * t
    b =  1.5 * t3 - 2.5 * t2 + 1.0
    c = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    d =  0.5 * t3 - 0.5 * t2
    return a * p0 + b * p1 + c * p2 + d * p3


def subdivide_open(indices, pts, k=4):
    """k extra points between each consecutive pair along an open polyline."""
    out = []
    n = len(indices)
    for i in range(n - 1):
        i0 = max(i - 1, 0); i1 = i; i2 = i + 1; i3 = min(i + 2, n - 1)
        p0, p1, p2, p3 = pts[indices[i0]], pts[indices[i1]], pts[indices[i2]], pts[indices[i3]]
        for s in range(1, k + 1):
            t = s / (k + 1)
            out.append(catmull_rom(p0, p1, p2, p3, t))
    return out


def subdivide_closed(indices, pts, k=3):
    """k extra points between each consecutive pair along a closed polyline."""
    out = []
    n = len(indices)
    for i in range(n):
        i0 = indices[(i - 1) % n]
        i1 = indices[i]
        i2 = indices[(i + 1) % n]
        i3 = indices[(i + 2) % n]
        p0, p1, p2, p3 = pts[i0], pts[i1], pts[i2], pts[i3]
        for s in range(1, k + 1):
            t = s / (k + 1)
            out.append(catmull_rom(p0, p1, p2, p3, t))
    return out


def bilinear_grid(corners, nu, nv):
    """Bilinear interpolation grid inside the quad corners=(p00,p10,p11,p01).
    nu, nv: interior point counts along each axis (excluding corners)."""
    p00, p10, p11, p01 = corners
    pts = []
    for j in range(1, nv + 1):
        v = j / (nv + 1)
        for i in range(1, nu + 1):
            u = i / (nu + 1)
            top = (1 - u) * p00 + u * p10
            bot = (1 - u) * p01 + u * p11
            pts.append((1 - v) * top + v * bot)
    return pts


def bary_triangle(corners, n_layers):
    """Barycentric sampling inside a triangle (corners=(p0, p1, p2)).
    Generates an N(N+1)/2-2-N point set in the interior (excluding edges)."""
    p0, p1, p2 = corners
    pts = []
    for i in range(1, n_layers):
        for j in range(1, n_layers - i):
            a = i / n_layers
            b = j / n_layers
            c = 1.0 - a - b
            if c <= 0: continue
            pts.append(a * p0 + b * p1 + c * p2)
    return pts


def midpoint(a, b, t=0.5):
    return (1 - t) * a + t * b


def main():
    canon = np.load(CANON)                         # [98, 2]
    pts98 = canon.astype(np.float32)
    virtuals = []

    def add(p_list):
        for p in p_list:
            virtuals.append(np.asarray(p, dtype=np.float32))

    # --- 1) Face contour subdivision (0..32) ---
    add(subdivide_open(list(range(33)), pts98, k=3))

    # --- 2) Brow loops (33..41, 42..50) ---
    add(subdivide_closed(list(range(33, 42)), pts98, k=2))
    add(subdivide_closed(list(range(42, 51)), pts98, k=2))

    # --- 3) Nose bridge (51..54) ---
    add(subdivide_open([51, 52, 53, 54], pts98, k=3))
    # nose bottom (55..59)
    add(subdivide_open([55, 56, 57, 58, 59], pts98, k=2))
    # nose bridge-to-bottom connections
    add(subdivide_open([54, 57], pts98, k=4))
    add(subdivide_open([54, 55], pts98, k=4))
    add(subdivide_open([54, 59], pts98, k=4))

    # --- 4) Eye loops (60..67, 68..75) ---
    add(subdivide_closed(list(range(60, 68)), pts98, k=2))
    add(subdivide_closed(list(range(68, 76)), pts98, k=2))

    # --- 5) Mouth loops ---
    add(subdivide_closed(list(range(76, 88)), pts98, k=2))  # outer
    add(subdivide_closed(list(range(88, 96)), pts98, k=2))  # inner

    # --- 6) Forehead grid: between top contour arc and brows ---
    # Top of forehead = lerp upward from brows
    brow_top_L  = pts98[33:42].mean(axis=0)
    brow_top_R  = pts98[42:51].mean(axis=0)
    forehead_top_L = pts98[0] + (brow_top_L - pts98[0]) * 0.0
    forehead_top_R = pts98[32] + (brow_top_R - pts98[32]) * 0.0
    # Two virtual rows above brows
    for row in range(1, 4):
        t = row / 4  # 0.25, 0.5, 0.75
        for i in range(7):
            u = i / 6
            # vertical: lerp from contour-top to brow
            cl = (1 - u) * pts98[0] + u * pts98[32]  # ~at contour top
            bl = (1 - u) * pts98[33] + u * pts98[50]  # ~at brow line
            p = (1 - t) * cl + t * bl
            virtuals.append(p.astype(np.float32))

    # --- 7) Left cheek interior (between left contour 3..12, left brow tail, left eye, nose-left) ---
    # Use triangle sampling: corners = left-contour-mid, left-eye-outer, mouth-left
    for layer_t in [0.25, 0.5, 0.75]:
        for k in range(1, 8):
            u = k / 8
            # along contour 4..12
            ci = int(4 + u * 8)
            ci = min(ci, 12)
            base = pts98[ci]
            # towards eye/cheek center
            target = midpoint(pts98[60], pts98[55], 0.4) if k < 4 else \
                     midpoint(pts98[76], pts98[55], 0.5)
            virtuals.append((base + (target - base) * layer_t).astype(np.float32))

    # --- 8) Right cheek interior (mirror) ---
    for layer_t in [0.25, 0.5, 0.75]:
        for k in range(1, 8):
            u = k / 8
            ci = int(28 - u * 8)
            ci = max(ci, 20)
            base = pts98[ci]
            target = midpoint(pts98[72], pts98[59], 0.4) if k < 4 else \
                     midpoint(pts98[82], pts98[59], 0.5)
            virtuals.append((base + (target - base) * layer_t).astype(np.float32))

    # --- 9) Philtrum / upper lip area: between nose 55..59 and outer mouth top ---
    nose_bottom = (pts98[55] + pts98[57] + pts98[59]) / 3
    mouth_top = (pts98[78] + pts98[82] + pts98[76]) / 3   # approx top-center of outer lips
    for i in range(8):
        for j in range(3):
            u = i / 7
            v = (j + 0.5) / 3.5
            left  = (1 - u) * pts98[55] + u * pts98[59]
            right = (1 - u) * pts98[76] + u * pts98[82]
            p = (1 - v) * left + v * right
            virtuals.append(p.astype(np.float32))

    # --- 10) Below mouth / chin grid ---
    chin_top_L = pts98[76]; chin_top_R = pts98[82]
    chin_bot_L = pts98[12]; chin_bot_R = pts98[20]
    for j in range(1, 5):
        v = j / 5
        for i in range(8):
            u = i / 7
            top = (1 - u) * chin_top_L + u * chin_top_R
            bot = (1 - u) * chin_bot_L + u * chin_bot_R
            virtuals.append(((1 - v) * top + v * bot).astype(np.float32))

    # --- 11) Inner mouth interior ---
    add(bary_triangle((pts98[88], pts98[92], (pts98[88] + pts98[92]) / 2 + np.array([0, 0.005])), 3))

    # --- 12) Around each eye (between brow and eye) ---
    for eye_idx, brow_idx_range in [
        (range(60, 68), range(33, 42)),
        (range(68, 76), range(42, 51)),
    ]:
        eye_pts = pts98[list(eye_idx)]
        brow_pts = pts98[list(brow_idx_range)]
        for j in range(3):
            v = (j + 1) / 4
            for ep, bp in zip(eye_pts, brow_pts):
                virtuals.append(((1 - v) * bp + v * ep).astype(np.float32))
        # And below the eye: connect to upper cheek line
        eye_bottom = eye_pts[4:7]
        cheek_anchor_L = pts98[5]; cheek_anchor_R = pts98[27]
        for ep in eye_bottom:
            for v in [0.3, 0.6]:
                target = cheek_anchor_L if ep[0] < 0.5 else cheek_anchor_R
                virtuals.append(((1 - v) * ep + v * target).astype(np.float32))

    # --- 13) Nose detail ---
    # Spine of the nose finer
    add(subdivide_open([51, 52, 53, 54, 57], pts98, k=4))
    # Nostril wings
    for side in ['L', 'R']:
        idx = 55 if side == 'L' else 59
        nose_corner = pts98[idx]
        for v in [0.3, 0.6]:
            virtuals.append(((1 - v) * nose_corner + v * pts98[57]).astype(np.float32))

    virtuals = np.stack(virtuals).astype(np.float32)
    all_pts = np.vstack([pts98, virtuals]).astype(np.float32)   # [N, 2]
    print(f"canonical 98 + virtual {len(virtuals)} = total {len(all_pts)}")

    # =================== TPS transfer matrix ===================
    # Solve TPS system so that, for current frame WFLW Y, dense pos = T @ Y.

    N = 98
    M = len(all_pts)

    # Kernel between canonical control points: K[i,j] = phi(||xi - xj||)
    def phi(r):
        # r^2 * log(r), with r=0 mapped to 0
        out = np.where(r > 1e-12, r * r * np.log(r + 1e-12), 0.0)
        return out

    Xc = pts98                                # [N, 2]
    diffs = Xc[:, None, :] - Xc[None, :, :]   # [N, N, 2]
    K = phi(np.linalg.norm(diffs, axis=2))     # [N, N]
    P = np.hstack([np.ones((N, 1)), Xc])        # [N, 3]

    # Build [K P; Pt 0] block
    top = np.hstack([K, P])
    bot = np.hstack([P.T, np.zeros((3, 3))])
    L = np.vstack([top, bot])                   # [N+3, N+3]
    # add small ridge for stability
    L_reg = L + 1e-7 * np.eye(N + 3)
    L_inv_left = np.linalg.solve(L_reg, np.vstack([np.eye(N), np.zeros((3, N))]))   # [N+3, N]

    # For each dense point p_j, build row [phi(p_j - x_i) for all i, 1, p_j_x, p_j_y]
    diffs2 = all_pts[:, None, :] - Xc[None, :, :]   # [M, N, 2]
    K_pj = phi(np.linalg.norm(diffs2, axis=2))      # [M, N]
    P_pj = np.hstack([np.ones((M, 1)), all_pts])     # [M, 3]
    rows = np.hstack([K_pj, P_pj])                   # [M, N+3]

    T = rows @ L_inv_left                            # [M, N]
    # Sanity: T @ Xc should equal all_pts (since canonical IS the canonical)
    recon = T @ Xc
    err = np.max(np.abs(recon - all_pts))
    print(f"TPS reconstruction max err: {err:.6e}  (should be tiny)")

    # =================== triangle topology ===================
    # Delaunay over the canonical positions. Topology stays valid across faces
    # since the WFLW landmark connectivity is roughly preserved.
    from scipy.spatial import Delaunay
    tri = Delaunay(all_pts).simplices                # [T, 3]
    print(f"Delaunay triangles: {len(tri)}")

    # Filter triangles whose any edge crosses forbidden anatomical edges.
    FORBID = []
    for i in range(60, 67): FORBID.append((i, i + 1))
    FORBID.append((67, 60))
    for i in range(68, 75): FORBID.append((i, i + 1))
    FORBID.append((75, 68))
    for i in range(76, 87): FORBID.append((i, i + 1))
    FORBID.append((87, 76))
    for i in range(88, 95): FORBID.append((i, i + 1))
    FORBID.append((95, 88))

    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    good = []
    for t in tri:
        edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
        bad = False
        for (a, b) in edges:
            for (c, d) in FORBID:
                if a == c or a == d or b == c or b == d: continue
                if seg_intersect(all_pts[a], all_pts[b], all_pts[c], all_pts[d]):
                    bad = True; break
            if bad: break
        if not bad: good.append(t.tolist())
    print(f"after forbid-crossing filter: {len(good)} triangles")

    # ============ save ============
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n_total": int(M),
        "n_real": 98,
        "n_virtual": int(M - 98),
        "canonical": all_pts.tolist(),
        "T": T.astype(np.float32).reshape(-1).tolist(),   # row-major [M*N]
        "triangles": good,
    }
    with open(OUT, "w") as f:
        json.dump(data, f)
    sz = OUT.stat().st_size
    print(f"wrote {OUT}  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
