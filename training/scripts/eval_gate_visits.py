"""Replay first-grant decisions against an independently labeled roster of gate visits."""
import argparse
import json
import operator

import numpy as np

from facex import CompactGallery
from eval_open_gallery import decision_metrics, validate_labels


def evaluate_visits(enrollment, enrolled_ids, visit_ids, visit_person_ids, probes,
                    frame_visit_ids, frame_numbers, min_score, min_margin, confirmations=(1, 2)):
    """A visit ends at the first ID accepted on the requested number of consecutive frames."""
    enrolled_ids, labels = validate_labels(enrollment, enrolled_ids, len(visit_ids), visit_person_ids)
    visit_ids, frame_ids, numbers = map(np.asarray, (visit_ids, frame_visit_ids, frame_numbers))
    if visit_ids.ndim != 1 or visit_ids.dtype.kind not in "iuUS":
        raise ValueError("Use integer or string visit IDs")
    if len(np.unique(visit_ids)) != len(visit_ids):
        raise ValueError("Visit IDs must be unique in the roster")
    if frame_ids.shape != (len(probes),) or frame_ids.dtype.kind != visit_ids.dtype.kind:
        raise ValueError("Need one matching visit ID per probe")
    if numbers.shape != frame_ids.shape or numbers.dtype.kind not in "iu" or np.any(numbers < 0):
        raise ValueError("Need one nonnegative integer frame number per probe")
    policies = sorted(set(map(operator.index, confirmations)))
    if not policies or policies[0] < 1:
        raise ValueError("Confirmation counts must be positive")

    roster = {key: i for i, key in enumerate(visit_ids.tolist())}
    try:
        visits = np.array([roster[key] for key in frame_ids.tolist()], dtype=np.intp)
    except KeyError as exc:
        raise ValueError("A frame belongs to a visit outside the roster") from exc
    order = np.lexsort((numbers, visits))
    same_visit = visits[order[1:]] == visits[order[:-1]]
    if np.any(same_visit & (numbers[order[1:]] == numbers[order[:-1]])):
        raise ValueError("Duplicate frame number in one visit")

    gallery = CompactGallery(enrollment)
    decisions = gallery.identify(probes, min_score=min_score, min_margin=min_margin)
    results = {}
    for required in policies:
        grants = np.full(len(visit_ids), -1, dtype=np.intp)
        previous_visit, previous_id, previous_number, run = -1, -1, -1, 0
        for i in order:
            visit, who, number = visits[i], decisions[i], int(numbers[i])
            if grants[visit] >= 0:
                continue
            consecutive = (visit == previous_visit and number == previous_number + 1
                           and who == previous_id)
            if who < 0:
                run = 0
            else:
                run = run + 1 if consecutive else 1
                if run == required:
                    grants[visit] = who
            previous_visit, previous_id, previous_number = visit, who, number
        metrics = decision_metrics(enrolled_ids, labels, grants)
        metrics["known_visits"] = metrics.pop("known_probes")
        metrics["unknown_visits"] = metrics.pop("unknown_probes")
        results[str(required)] = metrics
    return dict(gallery_size=len(enrolled_ids), visits=len(visit_ids), probe_frames=len(probes),
                visits_without_frames=len(visit_ids)-len(np.unique(visits)), min_score=min_score,
                min_margin=min_margin, consecutive_confirmations=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--min-score", required=True, type=float)
    parser.add_argument("--min-margin", required=True, type=float)
    parser.add_argument("--confirmations", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as data:
        keys = ("enrollment", "enrolled_ids", "visit_ids", "visit_person_ids", "probes",
                "frame_visit_ids", "frame_numbers")
        result = evaluate_visits(*(data[k] for k in keys), args.min_score, args.min_margin,
                                 args.confirmations)
    print(json.dumps(result, allow_nan=False, indent=2))
