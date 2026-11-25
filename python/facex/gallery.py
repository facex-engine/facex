"""Exact cosine search against a fixed enrollment matrix."""
import operator

import numpy as np

from .core import EMBEDDING_DIM


def _unit_rows(values):
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != EMBEDDING_DIM or raw.dtype.kind not in "iuf":
        raise ValueError("Expected a matrix of 512 dimensional real vectors")
    with np.errstate(over="ignore"):
        rows = raw.astype(np.float32, copy=False).astype(np.float64, order="C")
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite embedding")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, norms, out=rows, where=norms != 0)
    return rows


def _select(row, k):
    if k == 1:
        return np.array([row.argmax()])
    if k == len(row):
        return np.argsort(-row, kind="stable")
    cutoff = np.partition(row, len(row) - k)[-k]
    above = np.flatnonzero(row > cutoff)
    ties = np.flatnonzero(row == cutoff)[:k - len(above)]
    candidates = np.concatenate((above, ties))
    return candidates[np.lexsort((candidates, -row[candidates]))]


# Conservative absolute score error for unit vectors on CPU with IEEE arithmetic.
# Factor two at the cutoff covers errors in both the kth and competing scores.
def _cutoff_slack(dtype):
    unit = float(np.finfo(dtype).eps) / 2
    operations = 2 * EMBEDDING_DIM + 4
    return 2 * operations * unit / (1 - operations * unit)


_FP32_SLACK = _cutoff_slack(np.float32)
_FP64_SLACK = _cutoff_slack(np.float64)


def _finalists(rough, k, slack, exceptional=None):
    cutoff = rough.max() if k == 1 else np.partition(rough, len(rough) - k)[-k]
    mask = rough >= cutoff - slack
    if exceptional is not None:
        mask |= exceptional
    return np.flatnonzero(mask)


def _rank(rows, query, candidates, k, norms=None):
    exact = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), 256):
        take = candidates[start:start + 256]
        values = rows[take]
        if norms is not None:
            values = values.astype(np.float64)
            values /= norms[take, None]
        # Identical contiguous rows must have the same reduction order, even at
        # a BLAS tile boundary or when scored alone at the end of a shortlist.
        np.multiply(values, query, out=values)
        exact[start:start + len(take)] = np.sum(values, axis=1)
    np.clip(exact, -1., 1., out=exact)
    selected = _select(exact, k)
    return candidates[selected], exact[selected]


class Gallery:
    """Enrollment row number is the stable ID. Equal scores prefer the earlier row."""
    def __init__(self, embeddings):
        self._rows = _unit_rows(embeddings)
        if not len(self._rows):
            raise ValueError("Empty enrollment matrix")
        if not np.all(np.any(self._rows, axis=1)):
            raise ValueError("Enrollment vectors must have a direction")

    def search(self, queries, k=1):
        """Return ID and cosine matrices of shape (query count, k)."""
        k = operator.index(k)
        if not 1 <= k <= len(self._rows):
            raise ValueError("k must be within the enrollment size")
        queries = _unit_rows(queries)
        ids = np.empty((len(queries), k), dtype=np.intp)
        scores = np.empty((len(queries), k), dtype=np.float64)
        # Bound temporary score rows independently of the submitted batch size.
        block = max(1, (1 << 20) // (8 * len(self._rows)))
        for start in range(0, len(queries), block):
            values = queries[start:start + block] @ self._rows.T
            np.clip(values, -1., 1., out=values)
            for i, row in enumerate(values, start):
                if not queries[i].any():
                    ids[i], scores[i] = np.arange(k), 0.
                    continue
                candidates = _finalists(row, k, _FP64_SLACK)
                ids[i], scores[i] = _rank(self._rows, queries[i], candidates, k)
        return ids, scores

    def identify(self, queries, *, min_score, min_margin):
        """Return enrollment row IDs, or -1 when score or separation is insufficient.

        Thresholds here are inclusive. Equal best scores always require abstention.
        Use one row per identity when interpreting margins as identity separation.
        """
        if not -1 <= min_score <= 1 or not 0 <= min_margin <= 2:
            raise ValueError("Invalid cosine score or margin threshold")
        ids, values = self.search(queries, min(2, len(self._rows)))
        accepted = (values[:, 0] >= min_score) & np.any(np.asarray(queries, dtype=np.float32), axis=1)
        if values.shape[1] == 2:
            gap = values[:, 0] - values[:, 1]
            accepted &= (gap > 0) & (gap >= min_margin)
        elif min_margin > 0:
            accepted[:] = False
        return np.where(accepted, ids[:, 0], -1)


class CompactGallery(Gallery):
    """Single-query oriented gallery. Store original float32 rows and float64 norms."""

    def __init__(self, embeddings):
        raw = np.asarray(embeddings)
        if raw.ndim != 2 or raw.shape[1] != EMBEDDING_DIM or raw.dtype.kind not in "iuf":
            raise ValueError("Expected a matrix of 512 dimensional real vectors")
        if not len(raw):
            raise ValueError("Empty enrollment matrix")
        with np.errstate(over="ignore"):
            self._rows = raw.astype(np.float32, order="C", copy=True)
        self._norms = np.empty(len(raw), dtype=np.float64)
        for start in range(0, len(raw), 256):
            chunk = self._rows[start:start + 256].astype(np.float64)
            self._norms[start:start + len(chunk)] = np.linalg.norm(chunk, axis=1)
        if not np.isfinite(self._norms).all() or np.any(self._norms == 0):
            raise ValueError("Enrollment vectors must be finite and have a direction")
        # A conservative normal range leaves room for dot-product partial sums.
        self._exceptional = (self._norms < 1e-20) | (self._norms > 1e20)

    def search(self, queries, k=1):
        k = operator.index(k)
        if not 1 <= k <= len(self._rows):
            raise ValueError("k must be within the enrollment size")
        queries = _unit_rows(queries)
        ids = np.empty((len(queries), k), dtype=np.intp)
        scores = np.empty((len(queries), k), dtype=np.float64)
        for i, query in enumerate(queries):
            if not query.any():
                ids[i], scores[i] = np.arange(k), 0.
                continue
            # For ordinary rows, sum(abs(a*b)) / norm(a) <= 1 (Cauchy-Schwarz).
            # The FP32 dot error is bounded by gamma(2D), plus query conversion.
            # Exceptional rows always go through the float64 pass.
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                rough = (self._rows @ query.astype(np.float32)).astype(np.float64)
                rough /= self._norms
            rough[self._exceptional] = -np.inf
            candidates = _finalists(rough, k, _FP32_SLACK, self._exceptional)
            ids[i], scores[i] = _rank(self._rows, query, candidates, k, self._norms)
        return ids, scores


def enrollment_template(embeddings, weights=None, *, mode="mean"):
    """Combine views of one identity. Return a unit vector and directional agreement.

    The medoid maximizes weighted cosine similarity to all included views.
    Agreement is the resultant length on the unit sphere and is not a probability.
    """
    rows = _unit_rows(embeddings)
    if not len(rows) or mode not in ("mean", "medoid"):
        raise ValueError("Expected at least one view and mean or medoid mode")
    weight = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight.shape != (len(rows),) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Expected one finite nonnegative weight per view")
    usable = np.any(rows, axis=1) & (weight > 0)
    if not usable.any():
        raise ValueError("No usable enrollment view")
    rows, weight = rows[usable], weight[usable]
    weight = weight / weight.max()
    resultant = (weight / weight.sum()) @ rows
    length = np.linalg.norm(resultant)
    if length <= 8 * np.finfo(np.float64).eps:
        raise ValueError("Views have no consistent direction")
    vector = resultant / length if mode == "mean" else rows[np.argmax(rows @ resultant)]
    return vector.astype(np.float32), float(min(length, 1.))
