"""
v3 dense face mesh template — flow-line rendering instead of triangulation.

Instead of triangulating ~600 points and drawing straight-line edges, we
organize the virtual points into a network of "flow lines" (UV-style
horizontal and vertical curves on the face surface). Each flow line is
drawn as a smooth Catmull-Rom spline through its points. This is what
makes the mesh in the reference picture look like curved threads instead
of a wireframe.

Output:
  D:/apps/facex/wasm/face_mesh_template_v3.json
  {
    canonical: [[x,y], ...],         # all M points in canonical [0,1]
    T: [float * M*N],                # TPS transfer matrix (row-major)
    flow_open:   [[i0, i1, ...], ...]   # open polylines (smooth open curve)
    flow_closed: [[i0, i1, ...], ...]   # closed loops (smooth closed curve)
  }
"""

import json
from pathlib import Path

import numpy as np


CANON = Path("D:/apps/facex/training/landmark/dense/canonical_98.npy")
OUT = Path("D:/apps/facex/wasm/face_mesh_template_v3.json")


def lerp(a, b, t):
    return (1 - t) * a + t * b


def main():
    pts98 = np.load(CANON).astype(np.float32)
    # We'll keep canonical points in a flat list, and build flow lines
    # referencing them by index. Real WFLW points = indices 0..97.

    all_pts = [pts98[i].copy() for i in range(98)]   # indices 0..97

    flow_open   = []   # list of lists of indices
    flow_closed = []

    def add_idx(p):
        all_pts.append(np.asarray(p, dtype=np.float32))
        return len(all_pts) - 1

    # ============ anatomical contours (real points) ============
    flow_open.append(list(range(33)))                 # face contour (open arc)
    flow_closed.append(list(range(33, 42)))           # left brow loop
    flow_closed.append(list(range(42, 51)))           # right brow loop
    flow_open.append([51, 52, 53, 54])                # nose bridge
    flow_open.append([55, 56, 57, 58, 59])            # nose bottom
    flow_closed.append(list(range(60, 68)))           # left eye
    flow_closed.append(list(range(68, 76)))           # right eye
    flow_closed.append(list(range(76, 88)))           # outer mouth
    flow_closed.append(list(range(88, 96)))           # inner mouth

    # ============ Forehead grid ============
    # 5 horizontal rows × 11 columns, between top-contour and brow line.
    face_h = pts98[16, 1] - min(pts98[33, 1], pts98[50, 1])
    forehead_rows = []      # list of lists of indices (one per row)
    forehead_cols = [[] for _ in range(11)]
    for j in range(1, 6):                                # 5 rows
        t_row = j / 6
        row = []
        for i in range(11):                              # 11 cols
            u = i / 10
            top = lerp(pts98[0], pts98[32], u) + np.array([0.0, -0.30 * face_h])
            if u <= 0.5:
                bot = lerp(pts98[33], pts98[41], u * 2)
            else:
                bot = lerp(pts98[42], pts98[50], (u - 0.5) * 2)
            p = lerp(top, bot, t_row)
            idx = add_idx(p)
            row.append(idx)
            forehead_cols[i].append(idx)
        forehead_rows.append(row)
    # add rows + columns as flow lines
    for r in forehead_rows: flow_open.append(r)
    for c in forehead_cols: flow_open.append(c)
    # connect column tops to the contour and brow tops for continuity
    for i, col in enumerate(forehead_cols):
        u = i / 10
        # top end -> roughly contour midpoint or arc
        if u <= 0.5:
            brow = 33 + int(u * 2 * 8)
        else:
            brow = 42 + int((u - 0.5) * 2 * 8)
        # extend column to brow at bottom (already covered by brow line)
        col.append(brow)

    # ============ Cheek grids ============
    # Left cheek: 7 rows × 6 cols inside quad (contour_3, contour_12, mouth_76, eye_60)
    def cheek_grid(corners, n_rows=7, n_cols=6, brow_anchor=None, eye_anchor=None):
        rows = []
        cols = [[] for _ in range(n_cols)]
        for j in range(1, n_rows + 1):
            v = j / (n_rows + 1)
            row = []
            for i in range(n_cols):
                u = (i + 0.5) / n_cols
                p_top = lerp(corners[0], corners[3], u)
                p_bot = lerp(corners[1], corners[2], u)
                p = lerp(p_top, p_bot, v)
                idx = add_idx(p)
                row.append(idx)
                cols[i].append(idx)
            rows.append(row)
        return rows, cols

    left_corners  = [pts98[3], pts98[12], pts98[76], pts98[60]]
    right_corners = [pts98[29], pts98[20], pts98[82], pts98[72]]
    lrows, lcols = cheek_grid(left_corners)
    rrows, rcols = cheek_grid(right_corners)
    for r in lrows: flow_open.append(r)
    for c in lcols: flow_open.append(c)
    for r in rrows: flow_open.append(r)
    for c in rcols: flow_open.append(c)

    # ============ Chin / below-mouth grid ============
    chin_rows = []
    chin_cols = [[] for _ in range(10)]
    for j in range(1, 6):
        v = j / 6
        row = []
        for i in range(10):
            u = i / 9
            top = lerp(pts98[76], pts98[82], u)
            bot = lerp(pts98[12], pts98[20], u)
            p = lerp(top, bot, v)
            idx = add_idx(p)
            row.append(idx)
            chin_cols[i].append(idx)
        chin_rows.append(row)
    for r in chin_rows: flow_open.append(r)
    for c in chin_cols: flow_open.append(c)

    # ============ Philtrum / upper lip area (between nose and mouth) ============
    phil_rows = []
    for j in range(1, 4):
        v = j / 4
        row = []
        for i in range(8):
            u = i / 7
            top = lerp(pts98[55], pts98[59], u)
            bot = lerp(pts98[76], pts98[82], u)
            p = lerp(top, bot, v)
            idx = add_idx(p)
            row.append(idx)
        phil_rows.append(row)
    for r in phil_rows: flow_open.append(r)

    # ============ Eyebrow→Eye ladder ============
    # Vertical "rungs" from each brow pt to corresponding eye pt
    for brow_range, eye_range in [(range(33, 42), range(60, 68)),
                                   (range(42, 51), range(68, 76))]:
        bs = list(brow_range); es = list(eye_range)
        # match: first 8 brow points to 8 eye points
        for bp_idx, ep_idx in zip(bs[:8], es):
            line = [bp_idx]
            for t in [0.33, 0.66]:
                line.append(add_idx(lerp(pts98[bp_idx], pts98[ep_idx], t)))
            line.append(ep_idx)
            flow_open.append(line)

    # ============ Below-eye ladder (eye_bottom -> upper cheek) ============
    cheek_anchor_L = pts98[5]; cheek_anchor_R = pts98[27]
    for eye_range, cheek_anchor in [(range(63, 67), cheek_anchor_L),
                                     (range(71, 75), cheek_anchor_R)]:
        for ep_idx in eye_range:
            line = [ep_idx]
            for t in [0.33, 0.66]:
                line.append(add_idx(lerp(pts98[ep_idx], cheek_anchor, t)))
            flow_open.append(line)

    # ============ Nasolabial fold (nose corner → mouth corner) ============
    for nose_idx, mouth_idx, bow_dir in [(55, 76, -1), (59, 82, 1)]:
        line = [nose_idx]
        for s in range(1, 6):
            t = s / 6
            p = lerp(pts98[nose_idx], pts98[mouth_idx], t)
            # bow slightly outward toward cheek for natural fold curve
            p = p + np.array([bow_dir * 0.006, 0.0])
            line.append(add_idx(p))
        line.append(mouth_idx)
        flow_open.append(line)

    # ============ Nose-tip rays ============
    for nose_target in [55, 56, 57, 58, 59]:
        line = [54]
        for t in [0.33, 0.66]:
            line.append(add_idx(lerp(pts98[54], pts98[nose_target], t)))
        line.append(nose_target)
        flow_open.append(line)

    # ============ Bridge → inner eye corners ============
    for eye_corner in [60, 68]:
        line = [51]
        for t in [0.33, 0.66]:
            line.append(add_idx(lerp(pts98[51], pts98[eye_corner], t)))
        line.append(eye_corner)
        flow_open.append(line)

    all_pts = np.stack(all_pts).astype(np.float32)
    M = len(all_pts); N = 98
    print(f"total points: {M}  (real 98 + virtual {M - 98})")
    print(f"flow lines: {len(flow_open)} open + {len(flow_closed)} closed")

    # ============ TPS transfer matrix ============
    def phi(r):
        return np.where(r > 1e-12, r * r * np.log(r + 1e-12), 0.0)

    Xc = pts98
    K = phi(np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2))
    P = np.hstack([np.ones((N, 1)), Xc])
    L = np.vstack([np.hstack([K, P]), np.hstack([P.T, np.zeros((3, 3))])])
    L_inv_left = np.linalg.solve(L + 1e-7 * np.eye(N + 3),
                                  np.vstack([np.eye(N), np.zeros((3, N))]))
    diffs = all_pts[:, None, :] - Xc[None, :, :]
    K_pj = phi(np.linalg.norm(diffs, axis=2))
    P_pj = np.hstack([np.ones((M, 1)), all_pts])
    T = np.hstack([K_pj, P_pj]) @ L_inv_left
    print(f"TPS recon err: {np.max(np.abs(T @ Xc - all_pts)):.2e}")

    # ============ Delaunay triangulation for m1-1-style mesh ============
    from scipy.spatial import Delaunay
    tri = Delaunay(all_pts).simplices

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

    good_tris = []
    for t in tri:
        edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
        bad = False
        for (a, b) in edges:
            for (c, d) in FORBID:
                if a == c or a == d or b == c or b == d: continue
                if seg_intersect(all_pts[a], all_pts[b], all_pts[c], all_pts[d]):
                    bad = True; break
            if bad: break
        if not bad: good_tris.append([int(t[0]), int(t[1]), int(t[2])])
    print(f"triangles: {len(tri)} -> {len(good_tris)} after forbid filter")

    # ============ Per-vertex Z (relative depth, 0=back, 1=forward) ============
    # Hand-crafted depths for each of the 98 WFLW points, based on anatomy.
    # The face is roughly a half-ellipsoid: nose tip is the most forward,
    # ears are the most back. Cheekbones and brow ridges sit forward of jaw.
    z98 = np.zeros(98, dtype=np.float32)
    # Face contour 0..32 (ear -> jaw -> chin -> jaw -> ear)
    contour_zs = []
    for i in range(33):
        # parabolic profile: peak forward at chin (idx 16), back at ears (0, 32)
        t = i / 32
        u = abs(t - 0.5) * 2          # 0 at center (chin), 1 at ears
        contour_zs.append(0.6 - 0.7 * (u ** 1.5))
    z98[:33] = contour_zs
    # Brows: between forehead and brow ridge — fairly forward
    z98[33:42] = 0.55                 # left brow
    z98[42:51] = 0.55                 # right brow
    # Nose bridge 51..54: progressively forward to the tip
    z98[51] = 0.65; z98[52] = 0.75; z98[53] = 0.85; z98[54] = 0.95
    # Nose bottom 55..59 (nostrils, tip, nostrils)
    z98[55] = 0.80; z98[56] = 0.90; z98[57] = 1.00; z98[58] = 0.90; z98[59] = 0.80
    # Eyes: in eye sockets, slightly recessed from cheekbones
    z98[60:68] = 0.55                 # left eye loop
    z98[68:76] = 0.55                 # right eye loop
    # Mouth outer 76..87 — protrudes a bit
    z98[76:88] = 0.65
    # Mouth inner 88..95 — recessed (mouth opening)
    z98[88:96] = 0.45
    # Pupils
    z98[96] = 0.60; z98[97] = 0.60

    # Add a tiny per-region offset to differentiate features
    # (helps shading look more 3D)
    z98[15:18] += 0.05                # chin tip forward a tad
    z98[60:68] -= 0.05                # eyes deeper
    z98[68:76] -= 0.05

    # Derive Z for all virtual points using the same TPS transfer:
    # z_dense = T @ z98
    z_dense = (T @ z98.reshape(-1, 1)).reshape(-1).astype(np.float32)
    print(f"z range: 98 control [{z98.min():.2f}..{z98.max():.2f}], dense [{z_dense.min():.2f}..{z_dense.max():.2f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "version": 3,
        "n_total": int(M),
        "n_real": 98,
        "canonical": all_pts.tolist(),
        "z_canonical": z_dense.tolist(),
        "T": T.astype(np.float32).reshape(-1).tolist(),
        "flow_open": flow_open,
        "flow_closed": flow_closed,
        "triangles": good_tris,
    }, open(OUT, "w"))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
