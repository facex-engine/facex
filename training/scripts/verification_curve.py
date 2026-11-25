"""Empirical verification metrics from held-out, labeled pair scores."""
import argparse
import json
import math

import numpy as np


def _pairs(scores, same):
    scores, same = np.asarray(scores), np.asarray(same)
    if scores.ndim != 1 or scores.shape != same.shape or scores.dtype.kind not in "iuf":
        raise ValueError("Expected equal length vectors of real scores and binary labels")
    if not np.isfinite(scores).all() or not np.isin(same, [0, 1]).all():
        raise ValueError("Invalid score or label")
    positives = int(np.count_nonzero(same))
    negatives = len(same) - positives
    if not positives or not negatives:
        raise ValueError("Need genuine and impostor pairs")
    return scores, same.astype(bool), positives, negatives


def evaluate_threshold(scores, same, threshold):
    """Evaluate a frozen threshold on a separate set with independent pairs."""
    scores, same, positives, negatives = _pairs(scores, same)
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Expected a finite threshold or None for reject all")
    accepted = scores >= threshold if threshold is not None else np.zeros_like(same)
    fp = int(np.count_nonzero(accepted & ~same))
    fn = int(np.count_nonzero(~accepted & same))
    return dict(false_accepts=fp, false_rejects=fn, far=fp/negatives, frr=fn/positives,
                balanced_accuracy=1 - .5 * (fp/negatives + fn/positives),
                negatives=negatives, positives=positives,
                zero_false_accept_upper95=-math.expm1(math.log(.05)/negatives) if fp == 0 else None)


def verification_curve(scores, same, target_far):
    scores, same, positives, negatives = _pairs(scores, same)
    if not 0 <= target_far <= 1:
        raise ValueError("Expected target FAR between zero and one")
    order = np.argsort(scores, kind="stable")[::-1]
    ranked = scores[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked)-1]
    tp = np.r_[0, np.cumsum(same[order], dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    tar, far = tp / positives, fp / negatives
    auc = np.sum(np.diff(far) * (tar[1:] + tar[:-1]) * .5)
    valid = np.flatnonzero(far <= target_far)
    best = valid[np.argmax(tp[valid])]
    gap = far + tar - 1
    right = int(np.searchsorted(gap, 0))
    left = right - 1
    eer = far[left] + (far[right]-far[left]) * (-gap[left]) / (gap[right]-gap[left])
    return dict(auc=float(auc), interpolated_eer=float(eer),
                threshold=float(ranked[ends[best-1]]) if best else None,
                false_accepts=int(fp[best]), true_accepts=int(tp[best]),
                negatives=negatives, positives=positives, tar=float(tar[best]),
                far=float(far[best]), far_resolution=1 / negatives)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", help="NPZ file with scores and same arrays")
    parser.add_argument("--far", required=True, type=float)
    parser.add_argument("--holdout", help="Independent NPZ to evaluate at the frozen threshold")
    args = parser.parse_args()
    with np.load(args.pairs) as data:
        result = verification_curve(data["scores"], data["same"], args.far)
    if args.holdout:
        with np.load(args.holdout) as data:
            result["holdout"] = evaluate_threshold(data["scores"], data["same"], result["threshold"])
    print(json.dumps(result))
