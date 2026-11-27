import unittest

import numpy as np

from facex import IdentityGallery


def oracle(rows, labels, queries, threshold, margin):
    rows = rows.astype('d')
    rows /= np.linalg.norm(rows, axis=1)[:, None]
    ids = np.unique(labels)
    results = []
    for q in queries.astype('d'):
        norm = np.linalg.norm(q[None], axis=1)[0]
        if not norm or (len(ids) == 1 and margin > 0):
            results.append(-1)
            continue
        cosine = np.clip(np.sum(rows * (q / norm), axis=1), -1, 1)
        scores = np.array([cosine[labels == i].max() for i in ids])
        order = np.argsort(-scores, kind='stable')
        gap = scores[order[0]] - scores[order[1]] if len(ids) > 1 else np.inf
        results.append(ids[order[0]] if scores[order[0]] >= threshold
                       and gap > 0 and gap >= margin else -1)
    return results


class IdentityGalleryTest(unittest.TestCase):
    def test_same_identity_ties_and_exact_thresholds(self):
        rows = np.eye(4, 512, dtype='f')
        labels = np.array([41, 7, 41, 99])
        rows[2] = rows[0]
        index = IdentityGallery(rows, labels)
        query = rows[[0]]
        np.testing.assert_array_equal(index.identify(query, min_score=1, min_margin=1), [41])
        np.testing.assert_array_equal(index.identify(query, min_score=1,
                                                   min_margin=np.nextafter(1., 2.)), [-1])
        rows[3] = rows[0]
        tied = IdentityGallery(rows, labels)
        np.testing.assert_array_equal(tied.identify(query, min_score=1, min_margin=0), [-1])
        np.testing.assert_array_equal(index.identify(query, min_score=1, min_margin=1), [41])

    def test_mixed_queries(self):
        rng = np.random.default_rng(28)
        data = rng.normal(size=(150, 512)).astype('f')
        data[0] *= 1e-35
        data[-1] *= 1e30
        labels = np.tile([8, 13, 99], 50)
        queries = np.vstack((data[::9], np.zeros((1,512), 'f'),
                             rng.normal(size=(29,512)).astype('f')))[:, ::-1]
        rows = data[:, ::-1]
        for caps in (1, 8):
            gallery = IdentityGallery(rows, labels, max_caps=caps)
            for threshold, margin in ((-1, 0), (.75, .05), (1, 0), (.1, .02)):
                kw = dict(min_score=threshold, min_margin=margin)
                expected = np.array(oracle(rows, labels, queries, threshold, margin))
                got = gallery.identify(queries, **kw)
                ids, resolved = gallery.screen(queries, **kw)
                np.testing.assert_array_equal(expected[resolved], ids[resolved])
                np.testing.assert_array_equal(expected, got)

    def test_unresolved_can_accept_or_reject(self):
        first, second, other = np.eye(3, 512, dtype="f")
        rows = np.stack((first, second, other))
        gallery = IdentityGallery(rows, [10, 10, 20], max_caps=1)
        # One cap contains two orthogonal views. Its witness misses one of them.
        kw = dict(min_score=.9, min_margin=.05)
        _, flags = gallery.screen(rows[:2], **kw)
        self.assertEqual(int(flags.sum()), 1)
        np.testing.assert_array_equal(gallery.identify(rows[:2], **kw), [10, 10])
        midpoint = ((first + second) / np.sqrt(2))[None]
        _, flags = gallery.screen(midpoint, **kw)
        self.assertFalse(flags[0])
        np.testing.assert_array_equal(gallery.identify(midpoint, **kw), [-1])

    def test_single_identity_and_empty_batch(self):
        row = np.eye(1, 512, dtype='f')
        gallery = IdentityGallery(row, [200])
        for threshold, margin, answer in ((1, 0, 200), (1, .01, -1)):
            kw = dict(min_score=threshold, min_margin=margin)
            np.testing.assert_array_equal(gallery.identify(row, **kw), [answer])
            np.testing.assert_array_equal(gallery.identify(row * 0, **kw), [-1])
            self.assertEqual(gallery.identify(row[:0], **kw).shape, (0,))
        for bad in ([-1], [2.5], [], [2**63]):
            with self.assertRaises(ValueError):
                IdentityGallery(row, bad)


if __name__ == '__main__':
    unittest.main()
