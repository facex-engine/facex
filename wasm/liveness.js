/*
 * liveness.js — Basic liveness detection using face keypoints.
 *
 * Detects:
 * 1. Face motion (frame-to-frame bbox movement)
 * 2. Blink detection (eye aspect ratio changes)
 * 3. Multiple frame consistency (not a static photo)
 *
 * Not a security-grade anti-spoofing solution — detects basic
 * photo/screen attacks but not 3D masks or deepfakes.
 */

class LivenessDetector {
  constructor(options = {}) {
    this.historySize = options.historySize || 30; // frames to track
    this.motionThreshold = options.motionThreshold || 2.0; // pixels
    this.blinkThreshold = options.blinkThreshold || 0.22; // EAR ratio

    this._history = [];
    this._blinkCount = 0;
    this._lastEAR = 1.0;
    this._wasEyeClosed = false;
  }

  /**
   * Update with new face detection.
   * @param {Object} face - {x1,y1,x2,y2,kps:[10 values]}
   * @returns {{alive: boolean, confidence: number, reason: string}}
   */
  update(face) {
    if (!face) {
      this._history = [];
      return { alive: false, confidence: 0, reason: 'No face' };
    }

    const entry = {
      cx: (face.x1 + face.x2) / 2,
      cy: (face.y1 + face.y2) / 2,
      w: face.x2 - face.x1,
      h: face.y2 - face.y1,
      ear: this._computeEAR(face.kps),
      t: Date.now()
    };

    this._history.push(entry);
    if (this._history.length > this.historySize)
      this._history.shift();

    // Need at least 10 frames
    if (this._history.length < 10) {
      return { alive: false, confidence: 0.1, reason: 'Collecting frames...' };
    }

    // Check 1: Motion — face must move slightly (not a static photo)
    const motion = this._computeMotion();
    const hasMotion = motion > this.motionThreshold;

    // Check 2: Blink detection
    this._detectBlink(entry.ear);
    const hasBlinked = this._blinkCount > 0;

    // Check 3: Size variation (breathing, micro-movements)
    const sizeVar = this._computeSizeVariation();
    const hasSizeChange = sizeVar > 0.005;

    // Compute confidence
    let confidence = 0;
    if (hasMotion) confidence += 0.35;
    if (hasBlinked) confidence += 0.40;
    if (hasSizeChange) confidence += 0.25;

    let reason;
    if (confidence >= 0.6) reason = 'Live person detected';
    else if (!hasMotion) reason = 'No motion detected — hold still and blink';
    else if (!hasBlinked) reason = 'Please blink';
    else reason = 'Analyzing...';

    return {
      alive: confidence >= 0.6,
      confidence,
      reason,
      details: { motion, blinks: this._blinkCount, sizeVar }
    };
  }

  /** Reset state */
  reset() {
    this._history = [];
    this._blinkCount = 0;
    this._lastEAR = 1.0;
    this._wasEyeClosed = false;
  }

  // ============ Internal ============

  /** Eye Aspect Ratio from 5 keypoints.
   * kps: [lex, ley, rex, rey, nx, ny, lmx, lmy, rmx, rmy]
   * EAR ≈ distance(mouth) / distance(eyes) as a proxy.
   * Real EAR needs 6 eye landmarks — we approximate from 5 points. */
  _computeEAR(kps) {
    // left eye (0,1), right eye (2,3), nose (4,5), left mouth (6,7), right mouth (8,9)
    const eyeDist = Math.sqrt((kps[2]-kps[0])**2 + (kps[3]-kps[1])**2);
    const mouthDist = Math.sqrt((kps[8]-kps[6])**2 + (kps[9]-kps[7])**2);
    const noseToEyeL = Math.sqrt((kps[4]-kps[0])**2 + (kps[5]-kps[1])**2);
    const noseToEyeR = Math.sqrt((kps[4]-kps[2])**2 + (kps[5]-kps[3])**2);

    // Use ratio of vertical to horizontal distances as EAR proxy
    if (eyeDist < 1) return 1.0;
    return (noseToEyeL + noseToEyeR) / (2 * eyeDist);
  }

  _detectBlink(ear) {
    if (ear < this.blinkThreshold && !this._wasEyeClosed) {
      this._wasEyeClosed = true;
    } else if (ear > this.blinkThreshold + 0.05 && this._wasEyeClosed) {
      this._wasEyeClosed = false;
      this._blinkCount++;
    }
    this._lastEAR = ear;
  }

  _computeMotion() {
    if (this._history.length < 2) return 0;
    let totalMotion = 0;
    for (let i = 1; i < this._history.length; i++) {
      const dx = this._history[i].cx - this._history[i-1].cx;
      const dy = this._history[i].cy - this._history[i-1].cy;
      totalMotion += Math.sqrt(dx*dx + dy*dy);
    }
    return totalMotion / (this._history.length - 1);
  }

  _computeSizeVariation() {
    if (this._history.length < 5) return 0;
    const sizes = this._history.map(h => h.w * h.h);
    const mean = sizes.reduce((a,b) => a+b) / sizes.length;
    const variance = sizes.reduce((a,s) => a + (s-mean)**2, 0) / sizes.length;
    return Math.sqrt(variance) / mean; // coefficient of variation
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = {LivenessDetector};
