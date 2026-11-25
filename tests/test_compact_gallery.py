"""Verify both gallery representations against exhaustive row-wise float64 cosine."""
import unittest

import numpy as np

from facex import CompactGallery, Gallery


def reference(rows, queries, k):
    rows = np.array(rows, dtype="d", order="C")
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    ids, values = [], []
    for query in np.array(queries, dtype="d", order="C"):
        norm = np.linalg.norm(query[None, :], axis=1)[0]
        if norm:
            query /= norm
        scores = np.clip(np.sum(rows * query, axis=1), -1., 1.)
        chosen = np.argsort(-scores, kind="stable")[:k]
        ids.append(chosen)
        values.append(scores[chosen])
    return np.array(ids), np.array(values)


class CompactGalleryTest(unittest.TestCase):
    def test_precision_range_and_cutoff(self):
        rng = np.random.default_rng(152)
        rows = rng.standard_normal((517,512)).astype("f")
        rows[:8] *= 1e-40
        rows[8:16] *= 1e32
        rows[16] = np.finfo("f").max
        rows[17] = -np.nextafter(np.float32(0), np.float32(1))
        rows[18:280] = rows[-1]
        rows[18:280,0] += rng.uniform(-2e-5,2e-5,262).astype("f")
        queries = np.vstack((rows[[0,9,16,17,18]], -rows[50], rows[-1]*0))
        for layout in (rows, np.asfortranarray(rows), rows[:, ::-1]):
            for k in (1, 5, 261, len(rows)):
                expected_ids, expected_scores = reference(layout, queries, k)
                for cls in (Gallery, CompactGallery):
                    ids, scores = cls(layout).search(queries,k)
                    np.testing.assert_array_equal(ids,expected_ids)
                    np.testing.assert_array_equal(scores,expected_scores)

    def test_identical_rows_and_boundary(self):
        rng = np.random.default_rng(21)
        row, query = rng.normal(size=(2,512)).astype("f")
        rows = np.tile(row,(1025,1))
        for cls in (Gallery, CompactGallery):
            gallery = cls(rows)
            ids,scores = gallery.search([query],5)
            np.testing.assert_array_equal(ids, [np.arange(5)])
            self.assertTrue((scores == scores[0,0]).all())
            self.assertEqual(gallery.identify([query],min_score=-1,min_margin=0)[0],-1)

        rows = np.vstack((row, query, -query))
        top, scores = reference(rows,[row],2)
        margin = scores[0,0]-scores[0,1]
        for cls in (Gallery, CompactGallery):
            gallery = cls(rows)
            for m, decision in ((margin,top[0,0]),(np.nextafter(margin,2.),-1)):
                self.assertEqual(gallery.identify([row],min_score=scores[0,0],min_margin=m)[0],decision)

    def test_snapshot_and_empty_queries(self):
        rows = np.eye(2,512,dtype="f")
        gallery = CompactGallery(rows)
        query = rows[1:].copy()
        rows[:] = 0
        self.assertEqual(gallery.search(query)[0][0,0],1)
        self.assertEqual(gallery.search(query[:0],2)[1].shape,(0,2))
        for bad in (rows, rows[:0], np.full_like(rows,np.inf), np.ones((1,511))):
            with self.assertRaises(ValueError):
                CompactGallery(bad)


if __name__ == "__main__":
    unittest.main()
