"""Evaluate rank-one open-set decisions at frozen score and margin thresholds."""
import argparse
import json
import math

import numpy as np

from facex import CompactGallery


def _macro(events, labels):
    _, group = np.unique(labels, return_inverse=True)
    return float(np.mean(np.bincount(group, weights=events) / np.bincount(group)))


def validate_labels(enrollment, enrolled_ids, probe_count, probe_ids):
    enrolled_ids, probe_ids = np.asarray(enrolled_ids), np.asarray(probe_ids)
    if enrolled_ids.ndim != 1 or len(enrolled_ids) != len(enrollment):
        raise ValueError("Need one identity per enrollment row")
    if probe_ids.ndim != 1 or len(probe_ids) != probe_count:
        raise ValueError("Need one identity per probe")
    kind = enrolled_ids.dtype.kind
    if not ((kind in "iu" and probe_ids.dtype.kind == kind) or
            (kind in "US" and probe_ids.dtype.kind == kind)):
        raise ValueError("Use matching integer or string identity types")
    if len(np.unique(enrolled_ids)) != len(enrolled_ids):
        raise ValueError("Pool multiple views before enrolling an identity")
    known = np.isin(probe_ids, enrolled_ids)
    if not known.any() or known.all():
        raise ValueError("Need both enrolled and unenrolled probes")
    return enrolled_ids, probe_ids


def decision_metrics(enrolled_ids, labels, decisions, independent=False):
    known = np.isin(labels, enrolled_ids)
    accepted = decisions >= 0
    correct = np.zeros(len(labels), dtype=bool)
    correct[accepted] = enrolled_ids[decisions[accepted]] == labels[accepted]
    unknown = ~known
    errors = int(accepted[unknown].sum())
    count = int(unknown.sum())
    result = dict(unknown_probes=count, false_identifications=errors, fpir=errors/count,
                  macro_fpir=_macro(accepted[unknown], labels[unknown]),
                  known_probes=int(known.sum()), correct_identifications=int(correct.sum()),
                  known_rejected=int((known & ~accepted).sum()),
                  known_misidentified=int((known & accepted & ~correct).sum()),
                  fnir=float((~correct[known]).mean()), macro_fnir=_macro(~correct[known], labels[known]))
    if independent and errors == 0:
        result["zero_false_id_upper95"] = -math.expm1(math.log(.05)/count)
    return result


def evaluate(enrollment, enrolled_ids, probes, probe_ids, min_score, min_margin, independent=False):
    enrolled_ids, labels = validate_labels(enrollment, enrolled_ids, len(probes), probe_ids)
    decisions = CompactGallery(enrollment).identify(probes, min_score=min_score, min_margin=min_margin)
    result = decision_metrics(enrolled_ids, labels, decisions, independent)
    return dict(result, gallery_size=len(enrollment), min_score=min_score, min_margin=min_margin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="NPZ containing enrollment, enrolled_ids, probes and probe_ids")
    parser.add_argument("--min-score", type=float, required=True)
    parser.add_argument("--min-margin", type=float, required=True)
    parser.add_argument("--independent-probes", action="store_true")
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        result = evaluate(*(data[k] for k in ("enrollment","enrolled_ids","probes","probe_ids")),
                          args.min_score, args.min_margin, args.independent_probes)
    print(json.dumps(result, allow_nan=False))
