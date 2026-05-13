"""
v2 dense face mesh template — denser, multi-resolution, anatomically refined.

Improvements over v1:
  - ~1200 total points (98 real + ~1100 virtual)
  - point density tuned per region (more on cheeks/forehead/around eyes/lips)
  - multi-resolution: 3 levels (300/600/1200 pts) using same control matrix
  - extra forced-edges (eyelid creases, nasolabial folds) so triangulation
    follows facial anatomy instead of cutting through it
  - clip virtual points to inside the face contour polygon
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


CANON = Path("D:/apps/facex/training/landmark/dense/canonical_98.npy")
OUT = Path("D:/apps/facex/wasm/face_mesh_template_v2.json")


# ============== curve / grid helpers ==============

def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t; t3 = t2 * t
    a = -0.5 * t3 + t2 - 0.5 * t
    b =  1.5 * t3 - 2.5 * t2 + 1.0
    c = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    d =  0.5 * t3 - 0.5 * t2
    return a * p0 + b * p1 + c * p2 + d * p3


def subdivide_open(idx, pts, k):
    out = []
    n = len(idx)
    for i in range(n - 1):
        p0 = pts[idx[max(i-1, 0)]]; p1 = pts[idx[i]]
        p2 = pts[idx[i+1]];          p3 = pts[idx[min(i+2, n-1)]]
        for s in range(1, k + 1):
            out.append(catmull_rom(p0, p1, p2, p3, s / (k + 1)))
    return out


def subdivide_closed(idx, pts, k):
    out = []
    n = len(idx)
    for i in range(n):
        p0 = pts[idx[(i-1) % n]]; p1 = pts[idx[i]]
        p2 = pts[idx[(i+1) % n]]; p3 = pts[idx[(i+2) % n]]
        for s in range(1, k + 1):
            out.append(catmull_rom(p0, p1, p2, p3, s / (k + 1)))
    return out


def point_in_polygon(p, poly):
    """Ray casting test."""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def main():
    pts98 = np.load(CANON).astype(np.float32)
    virtuals = []

    def add(p_list):
        for p in p_list:
            virtuals.append(np.asarray(p, dtype=np.float32))

    def lerp(a, b, t):
        return (1 - t) * a + t * b

    # --- contour subdivision: 5 between each (33 pts -> +160) ---
    add(subdivide_open(list(range(33)), pts98, k=5))

    # --- brows: 3 between each, both loops ---
    add(subdivide_closed(list(range(33, 42)), pts98, k=3))
    add(subdivide_closed(list(range(42, 51)), pts98, k=3))

    # --- nose: bridge 3 between, bottom 3 between, plus tip refinement ---
    add(subdivide_open([51, 52, 53, 54], pts98, k=4))
    add(subdivide_open([55, 56, 57, 58, 59], pts98, k=3))
    # bridge -> bottom (multiple connecting lines for sides)
    add(subdivide_open([54, 57], pts98, k=5))
    add(subdivide_open([54, 55], pts98, k=5))
    add(subdivide_open([54, 59], pts98, k=5))
    add(subdivide_open([52, 60], pts98, k=4))   # bridge to inner-eye L
    add(subdivide_open([52, 68], pts98, k=4))   # bridge to inner-eye R

    # --- eyes: tight loops with 3 between each ---
    add(subdivide_closed(list(range(60, 68)), pts98, k=3))
    add(subdivide_closed(list(range(68, 76)), pts98, k=3))

    # --- mouth outer + inner ---
    add(subdivide_closed(list(range(76, 88)), pts98, k=3))
    add(subdivide_closed(list(range(88, 96)), pts98, k=3))

    # Upper-lip mid filaments (philtrum hint)
    for u in np.linspace(0.2, 0.8, 5):
        add([lerp(pts98[78], pts98[88 + (1 if u > 0.5 else 0)], 0.5)])

    # --- Forehead grid (rows of points above brows) ---
    # 6 columns from contour 0 → contour 32, 4 rows above brows
    for j in range(1, 6):                          # 5 vertical layers
        t = j / 6
        # interpolate from "top contour" (using point 0 and 32) down to brow line
        for i in range(11):                        # 11 columns
            u = i / 10
            top = lerp(pts98[0], pts98[32], u)
            # brow line: smooth interpolation between brow tops (33 and 50, with middle)
            if u <= 0.5:
                bot = lerp(pts98[33], pts98[41], u * 2)
            else:
                bot = lerp(pts98[42], pts98[50], (u - 0.5) * 2)
            virtuals.append(lerp(top, bot, t).astype(np.float32))

    # --- Cheek grids (left & right): dense, 8x6 each ---
    # left cheek: bounded by contour 2..12, brow tail 33-34, eye 60-67, nose 55, mouth 76
    left_corners = [pts98[3], pts98[12], pts98[76], pts98[60]]    # rough quad
    for j in range(1, 8):     # 7 rows
        v = j / 8
        for i in range(1, 7): # 6 columns
            u = i / 7
            # bilinear inside quad
            p_top = lerp(left_corners[0], left_corners[3], u)
            p_bot = lerp(left_corners[1], left_corners[2], u)
            virtuals.append(lerp(p_top, p_bot, v).astype(np.float32))

    right_corners = [pts98[29], pts98[20], pts98[82], pts98[72]]
    for j in range(1, 8):
        v = j / 8
        for i in range(1, 7):
            u = i / 7
            p_top = lerp(right_corners[0], right_corners[3], u)
            p_bot = lerp(right_corners[1], right_corners[2], u)
            virtuals.append(lerp(p_top, p_bot, v).astype(np.float32))

    # --- Around-eye grids: between brow and eye + nasolabial side ---
    for eye_range, brow_range, cheek_anchor in [
        (range(60, 68), range(33, 42), pts98[5]),
        (range(68, 76), range(42, 51), pts98[27]),
    ]:
        eye_pts = pts98[list(eye_range)]
        brow_pts = pts98[list(brow_range)]
        for v in [0.25, 0.5, 0.75]:
            for ep, bp in zip(eye_pts, brow_pts):
                virtuals.append(lerp(bp, ep, v).astype(np.float32))
        # below-eye: between eye-bottom and upper-cheek
        for ep in eye_pts[3:7]:
            for v in [0.25, 0.5, 0.75]:
                virtuals.append(lerp(ep, cheek_anchor, v).astype(np.float32))

    # --- Below-mouth / chin grid: 8 cols × 5 rows ---
    for j in range(1, 6):
        v = j / 6
        for i in range(10):
            u = i / 9
            top = lerp(pts98[76], pts98[82], u)
            bot = lerp(pts98[12], pts98[20], u)
            virtuals.append(lerp(top, bot, v).astype(np.float32))

    # --- Upper lip / philtrum: between nose-bottom and outer mouth top ---
    for j in range(1, 4):
        v = j / 4
        for i in range(8):
            u = i / 7
            top = lerp(pts98[55], pts98[59], u)
            bot = lerp(pts98[76], pts98[82], u)
            virtuals.append(lerp(top, bot, v).astype(np.float32))

    # --- Nasolabial folds: parametric curves from nose corner to mouth corner ---
    for side_nose, side_mouth in [(55, 76), (59, 82)]:
        for s in range(1, 6):
            t = s / 6
            p = lerp(pts98[side_nose], pts98[side_mouth], t)
            # bow slightly outward (toward cheek)
            normal_dir = np.array([-1 if side_nose == 55 else 1, 0]) * 0.005
            virtuals.append((p + normal_dir).astype(np.float32))

    virtuals = np.stack(virtuals).astype(np.float32)

    # --- Clip virtual points to inside (extended) face polygon ---
    # WFLW contour doesn't include forehead; extend the polygon upward over
    # the brow line so forehead grid points are kept inside.
    contour_poly = pts98[:33].tolist()
    # forehead arc: from contour_32 -> over brows -> contour_0
    # Approximate the forehead top by lifting brow midpoint upward by 0.18 face-height
    face_h = pts98[16, 1] - min(pts98[33, 1], pts98[50, 1])
    forehead_top_R = (pts98[50] + np.array([0.0, -0.20 * face_h])).tolist()
    forehead_mid   = ((pts98[41] + pts98[42]) * 0.5 + np.array([0.0, -0.30 * face_h])).tolist()
    forehead_top_L = (pts98[33] + np.array([0.0, -0.20 * face_h])).tolist()
    contour_poly = contour_poly + [forehead_top_R, forehead_mid, forehead_top_L]

    keep = []
    for p in virtuals:
        if point_in_polygon((p[0], p[1]), contour_poly):
            keep.append(p)
    virtuals = np.stack(keep).astype(np.float32) if keep else virtuals[:0]
    print(f"after polygon clip: kept {len(virtuals)} of original (extended forehead)")

    all_pts = np.vstack([pts98, virtuals]).astype(np.float32)
    print(f"total points: {len(all_pts)} (98 real + {len(virtuals)} virtual)")

    # ============ TPS transfer matrix ============
    N = 98; M = len(all_pts)

    def phi(r):
        return np.where(r > 1e-12, r * r * np.log(r + 1e-12), 0.0)

    Xc = pts98
    K = phi(np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2))
    P = np.hstack([np.ones((N, 1)), Xc])
    L = np.vstack([np.hstack([K, P]), np.hstack([P.T, np.zeros((3, 3))])])
    L_reg = L + 1e-7 * np.eye(N + 3)
    L_inv_left = np.linalg.solve(L_reg, np.vstack([np.eye(N), np.zeros((3, N))]))

    diffs = all_pts[:, None, :] - Xc[None, :, :]
    K_pj = phi(np.linalg.norm(diffs, axis=2))
    P_pj = np.hstack([np.ones((M, 1)), all_pts])
    T = np.hstack([K_pj, P_pj]) @ L_inv_left
    err = np.max(np.abs(T @ Xc - all_pts))
    print(f"TPS recon max err: {err:.6e}")

    # ============ triangulation ============
    tri = Delaunay(all_pts).simplices

    # forbid crossings of eye/mouth contours
    FORBID = []
    for i in range(60, 67): FORBID.append((i, i + 1)); FORBID.append((67, 60))
    for i in range(68, 75): FORBID.append((i, i + 1)); FORBID.append((75, 68))
    for i in range(76, 87): FORBID.append((i, i + 1)); FORBID.append((87, 76))
    for i in range(88, 95): FORBID.append((i, i + 1)); FORBID.append((95, 88))

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
    print(f"triangles: {len(tri)} -> {len(good)} after constraint")

    # ============ multi-resolution decimation ============
    # Build 3 levels: low (300), mid (600), full (M).
    # For each level, keep all 98 real + a random subset of virtuals.
    # Triangles re-Delaunay on the kept subset.
    rng = np.random.default_rng(0)

    def make_level(n_target):
        if n_target >= M: return list(range(M)), good
        # keep all real (0..97), sample virtual indices uniformly
        n_virt = n_target - 98
        all_virt = list(range(98, M))
        # spread sampling to keep regional coverage: shuffle then take first n
        idx_virt = rng.permutation(all_virt)[:n_virt].tolist()
        keep_idx = sorted([i for i in range(98)] + idx_virt)
        keep_set = set(keep_idx)
        # remap old idx -> new idx
        remap = {old: new for new, old in enumerate(keep_idx)}
        new_pts = all_pts[keep_idx]
        # triangles on the kept subset (re-Delaunay)
        new_tri = Delaunay(new_pts).simplices
        # filter
        ng = []
        for t in new_tri:
            a, b, c = keep_idx[t[0]], keep_idx[t[1]], keep_idx[t[2]]
            bad = False
            edges = [(a, b), (b, c), (c, a)]
            for (ea, eb) in edges:
                for (ec, ed) in FORBID:
                    if ea == ec or ea == ed or eb == ec or eb == ed: continue
                    if seg_intersect(all_pts[ea], all_pts[eb], all_pts[ec], all_pts[ed]):
                        bad = True; break
                if bad: break
            if not bad: ng.append([int(t[0]), int(t[1]), int(t[2])])
        return keep_idx, ng

    levels = {}
    for name, n in [("low", 300), ("mid", 600), ("full", M)]:
        idx, tris = make_level(n)
        levels[name] = {"indices": idx, "triangles": tris}
        print(f"  level {name}: {len(idx)} pts, {len(tris)} triangles")

    # ============ save ============
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 2,
        "n_total": int(M), "n_real": 98,
        "canonical": all_pts.tolist(),
        "T": T.astype(np.float32).reshape(-1).tolist(),
        "levels": levels,
    }
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
