/*
 * align.js — 5-point face alignment for FaceX.
 *
 * Takes 5 keypoints from detector (left_eye, right_eye, nose, left_mouth, right_mouth)
 * and produces a 112x112 aligned face using affine transformation.
 *
 * Reference template: ArcFace standard 112x112 alignment targets.
 */

// ArcFace standard reference points for 112x112 output
const REF_POINTS = [
  [38.2946, 51.6963],  // left eye
  [73.5318, 51.5014],  // right eye
  [56.0252, 71.7366],  // nose
  [41.5493, 92.3655],  // left mouth
  [70.7299, 92.2041],  // right mouth
];

/**
 * Compute similarity transform matrix from src points to dst points.
 * Uses least-squares fit for 2D similarity (rotation + scale + translation).
 * Returns [a, b, tx, ty] where:
 *   x' = a*x - b*y + tx
 *   y' = b*x + a*y + ty
 */
function getSimilarityTransform(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i][0]; sy += src[i][1];
    dx += dst[i][0]; dy += dst[i][1];
  }
  sx /= n; sy /= n; dx /= n; dy /= n;

  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sxc = src[i][0] - sx, syc = src[i][1] - sy;
    const dxc = dst[i][0] - dx, dyc = dst[i][1] - dy;
    num1 += dxc * sxc + dyc * syc;
    num2 += dyc * sxc - dxc * syc;
    den += sxc * sxc + syc * syc;
  }

  const a = num1 / den;
  const b = num2 / den;
  const tx = dx - a * sx + b * sy;
  const ty = dy - b * sx - a * sy;

  return [a, b, tx, ty];
}

/**
 * Apply similarity transform to warp source image to 112x112 aligned face.
 *
 * @param {CanvasRenderingContext2D} srcCtx - source canvas context (video frame)
 * @param {number} srcW - source width
 * @param {number} srcH - source height
 * @param {Array<Array<number>>} kps - 5 keypoints [[x,y], ...] in source coords
 * @param {CanvasRenderingContext2D} dstCtx - destination 112x112 canvas context
 * @returns {ImageData} - 112x112 aligned face
 */
function alignFace(srcCtx, srcW, srcH, kps, dstCtx) {
  return alignFacePixels(srcCtx.getImageData(0, 0, srcW, srcH), kps, dstCtx);
}

function alignFacePixels(frame, kps, dstCtx) {
  const srcW = frame.width, srcH = frame.height;
  // We need inverse: for each dst pixel, find src pixel
  const [a, b, tx, ty] = getSimilarityTransform(kps, REF_POINTS);

  // Inverse transform: src = inverse(M) * dst
  // M = [a, -b, tx; b, a, ty]
  // M_inv = (1/det) * [a, b, -a*tx-b*ty; -b, a, b*tx-a*ty]
  const det = a * a + b * b;
  if (!(det > 0) || !Number.isFinite(det)) throw new RangeError('Degenerate landmark transform');
  const ai = a / det, bi = b / det;
  const txi = -(ai * tx + bi * ty);
  const tyi = (bi * tx - ai * ty);

  const epsilon = 64 * Number.EPSILON * Math.max(srcW, srcH);
  const dstData = dstCtx.createImageData(112, 112);
  const src = frame.data;
  const dst = dstData.data;

  for (let dy = 0; dy < 112; dy++) {
    for (let dx = 0; dx < 112; dx++) {
      // Map dst → src using inverse transform
      const sx = ai * dx + bi * dy + txi + epsilon;
      const sy = -bi * dx + ai * dy + tyi + epsilon;

      // Bilinear interpolation
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = Math.min(x0 + 1, srcW - 1), y1 = Math.min(y0 + 1, srcH - 1);
      const fx = sx - x0, fy = sy - y0;

      if (x0 >= 0 && x0 < srcW && y0 >= 0 && y0 < srcH) {
        const i00 = (y0 * srcW + x0) * 4;
        const i10 = (y0 * srcW + x1) * 4;
        const i01 = (y1 * srcW + x0) * 4;
        const i11 = (y1 * srcW + x1) * 4;
        const di = (dy * 112 + dx) * 4;

        for (let c = 0; c < 3; c++) {
          dst[di + c] = Math.round(
            src[i00 + c] * (1-fx) * (1-fy) +
            src[i10 + c] * fx * (1-fy) +
            src[i01 + c] * (1-fx) * fy +
            src[i11 + c] * fx * fy
          );
        }
        dst[di + 3] = 255;
      }
    }
  }

  dstCtx.putImageData(dstData, 0, 0);
  return dstData;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {alignFace, alignFacePixels};
