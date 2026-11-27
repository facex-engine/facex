"""Threshold identification with multiple enrollment rows per identity."""
import operator

import numpy as np

from .gallery import CompactGallery, _unit_rows


# Round the angular inputs outwards before evaluating the cap bound. Padding
# only the final score loses accuracy near a zero or pi angle.
_PAD = 1e-12


def _caps(rows, limit, rng):
    sample_ids = rng.choice(len(rows), min(len(rows), max(128, limit)), replace=False)
    sample = _unit_rows(rows[sample_ids])
    chosen = [0]
    nearest = sample @ sample[0]
    while len(chosen) < min(limit, len(sample)):
        nxt = nearest.argmin()
        if nearest[nxt] >= 1 - _PAD:
            break
        chosen.append(nxt)
        nearest = np.maximum(nearest, sample @ sample[nxt])
    axes = sample[chosen].copy()
    labels = (sample @ axes.T).argmax(axis=1)
    for i in range(len(axes)):
        members = sample[labels == i]
        axes[i] = members[(members @ members.mean(axis=0)).argmax()]
    cosines = np.ones(len(axes))
    for start in range(0, len(rows), 256):
        dots = _unit_rows(rows[start:start+256]) @ axes.T
        labels = dots.argmax(axis=1)
        np.minimum.at(cosines, labels, dots[np.arange(len(dots)), labels])
    return axes, np.clip(cosines - _PAD, -1, 1)


class IdentityGallery:
    """Fixed gallery of nonnegative int64 identity IDs with one or more views each.

    The identity score is the maximum cosine across its enrollment rows.
    The runner-up is always a different identity. A tied best score abstains.
    """

    def __init__(self, embeddings, identity_ids, *, max_caps=8):
        rows = np.asarray(embeddings)
        labels = np.asarray(identity_ids)
        cap_count = operator.index(max_caps)
        if rows.ndim != 2 or not len(rows):
            raise ValueError("Expected a nonempty enrollment matrix")
        if labels.shape != (len(rows),) or labels.dtype.kind not in "iu":
            raise ValueError("Expected one integer identity ID per enrollment row")
        if labels.min() < 0 or int(labels.max()) > np.iinfo(np.int64).max:
            raise ValueError("Identity IDs must fit in nonnegative int64")
        if cap_count < 1:
            raise ValueError("max_caps must be positive")

        order = np.argsort(labels, kind="stable")
        self._ids, counts = np.unique(labels, return_counts=True)
        self._ids = self._ids.astype(np.int64)
        self._galleries = []
        axes, cosines, starts = [], [], []
        offset = 0
        cap_offset = 0
        rng = np.random.default_rng(2026)
        for count in counts:
            group = rows[order[offset:offset+count]]
            self._galleries.append(CompactGallery(group))
            directions, limits = _caps(group, cap_count, rng)
            starts.append(cap_offset)
            axes.append(directions)
            cosines.append(limits)
            offset += count
            cap_offset += len(limits)
        self._starts = np.array(starts)
        self._axes = np.concatenate(axes)
        self._cos = np.concatenate(cosines)
        self._sin = np.sqrt((1 - self._cos) * (1 + self._cos))

    def screen(self, queries, *, min_score, min_margin):
        """Return (IDs, resolved mask). Resolved -1 is a rejection.

        Unresolved rows also carry -1 and require exact lookup. identify()
        performs that lookup automatically.
        """
        if not -1 <= min_score <= 1 or not 0 <= min_margin <= 2:
            raise ValueError("Invalid cosine score or margin threshold")
        unit = _unit_rows(queries)
        ids = np.full(len(unit), -1, dtype=np.int64)
        resolved = np.ones(len(unit), dtype=bool)
        for i, q in enumerate(unit):
            if not q.any() or (len(self._ids) == 1 and min_margin > 0):
                continue
            dots = self._axes @ q
            low = np.maximum.reduceat(dots - _PAD, self._starts)
            d = np.clip(dots + _PAD, -1, 1)
            upper = np.where(d >= self._cos, 1.,
                             d * self._cos + np.sqrt((1 - d) * (1 + d)) * self._sin)
            high = np.maximum.reduceat(upper + _PAD, self._starts)
            if high.max() < min_score:
                continue
            winner = low.argmax()
            high[winner] = -np.inf
            gap = low[winner] - high.max()
            if low[winner] >= min_score and gap > 0 and gap >= min_margin:
                ids[i] = self._ids[winner]
            else:
                resolved[i] = False
        return ids, resolved

    def identify(self, queries, *, min_score, min_margin):
        """Return exact threshold decisions, with -1 for rejection."""
        queries = np.asarray(queries)
        ids, resolved = self.screen(queries, min_score=min_score, min_margin=min_margin)
        remaining = np.flatnonzero(~resolved)
        if not len(remaining):
            return ids
        pending = queries[remaining]
        best = np.full(len(pending), -np.inf)
        second = best.copy()
        winner = np.empty(len(pending), dtype=np.int64)
        for identity, gallery in zip(self._ids, self._galleries):
            _, scores = gallery.search(pending, 1)
            score = scores[:, 0]
            improved = score > best
            second = np.where(improved, best, np.maximum(second, score))
            best = np.maximum(best, score)
            winner[improved] = identity
        accepted = (best >= min_score) & (best > second) & (best - second >= min_margin)
        ids[remaining[accepted]] = winner[accepted]
        return ids
