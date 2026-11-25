"""Independent numerical references for search, verification and guided geometry."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from facex import Gallery, enrollment_template, similarity
from facex.yunet import rectangle_shape, guided_crop, merge_faces

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/scripts"))
from pair_sampling import INDEX_RECORD, sample_index_pairs
from verification_curve import evaluate_threshold, verification_curve


class MathPipelineTest(unittest.TestCase):
    def test_scalar(self):
        z = np.zeros(512, dtype="f")
        for v in (np.finfo("f").max, np.finfo("f").tiny, np.nextafter(np.float32(0), np.float32(1))):
            a = np.full_like(z, v)
            self.assertEqual(similarity(a, a), 1)
            self.assertEqual(similarity(a, -a), -1)
            self.assertEqual(similarity(a, z), 0)
        for invalid in ([float("nan")]*512, [float("inf")]*512, [1]*511, ["1"]*512):
            with self.assertRaises(ValueError):
                similarity(z, invalid)

    def test_gallery(self):
        rng = np.random.default_rng(51)
        rows = rng.normal(size=(513, 512)).astype("f")
        rows[1:8] = rows[0]
        queries = rows[:257].copy()
        queries[1] = 0
        gallery = Gallery(rows)
        base = rows.astype("d")
        base /= np.linalg.norm(base, axis=1)[:, None]
        for k in (1, 4, 513):
            ids, values = gallery.search(queries, k)
            for q, found, scores in zip(queries, ids, values):
                q = q.astype("d")
                ref = np.clip(base @ (q / np.linalg.norm(q) if q.any() else q), -1, 1)
                np.testing.assert_array_equal(found, np.argsort(-ref, kind="stable")[:k])
                np.testing.assert_allclose(scores, ref[found], atol=2e-14)
        self.assertEqual(gallery.search(queries[:0])[0].shape, (0, 1))
        rows[:] = 0
        self.assertEqual(gallery.search(queries[:1])[0][0, 0], 0)
        with self.assertRaises(ValueError):
            gallery.search(queries, 514)
        with self.assertRaises(ValueError):
            Gallery(np.zeros((1,512)))

    def test_identify(self):
        a,b = np.eye(2,512)
        gallery = Gallery([a,b])
        ids = gallery.identify([a,b,a+b,-a,a*0], min_score=.3, min_margin=.1)
        np.testing.assert_array_equal(ids,[0,1,-1,-1,-1])
        self.assertEqual(Gallery([a,a]).identify([a], min_score=0, min_margin=0)[0],-1)
        self.assertEqual(Gallery([a]).identify([a], min_score=1, min_margin=.1)[0],-1)

    def test_enrollment(self):
        rng = np.random.default_rng(82)
        rows = rng.normal(size=(17,512)).astype("f")
        units = rows.astype("d")
        units /= np.linalg.norm(units,axis=1)[:,None]
        weights = rng.uniform(0,1,17)
        medoid = np.argmax((units @ units.T) @ weights)
        for scale in (1.,1e200,1e-200):
            result, coherence = enrollment_template(rows,weights*scale,mode="medoid")
            np.testing.assert_array_equal(result,units[medoid].astype("f"))
            self.assertAlmostEqual(coherence,np.linalg.norm(np.average(units,axis=0,weights=weights)))
        pooled, _ = enrollment_template([rows[0],rows[1],rows[2]*0],[1,0,1])
        np.testing.assert_array_equal(pooled,units[0].astype("f"))
        with self.assertRaises(ValueError):
            enrollment_template([rows[0],-rows[0]])

    def test_verification(self):
        rng = np.random.default_rng(126)
        for _ in range(100):
            a = rng.integers(-4, 5, 23)
            b = rng.integers(-4, 5, 29)
            d = a[:, None] - b
            expected = (np.mean(d > 0) + .5*np.mean(d == 0))
            for target in (0, .1, 1):
                out = verification_curve(np.r_[a,b], np.r_[np.ones(len(a)),np.zeros(len(b))], target)
                self.assertAlmostEqual(out["auc"], expected)
                self.assertLessEqual(out["far"], target)
                candidates = [(sum(a >= t), -sum(b >= t)) for t in range(-4, 6)
                              if np.mean(b >= t) <= target]
                self.assertEqual((out["true_accepts"], -out["false_accepts"]), max(candidates))
                self.assertAlmostEqual(out["far_resolution"], 1/len(b))
        for pos, neg, eer in (([1], [0], 0), ([0], [1], 1), ([1], [1], .5)):
            out = verification_curve(pos+neg, [1,0], 0)
            self.assertAlmostEqual(out["interpolated_eer"], eer)
        for s, labels in (([1,2],[0,0]), ([np.nan,1],[0,1]), ([1,2],[1,2])):
            with self.assertRaises(ValueError):
                verification_curve(s, labels, 0)

    def test_frozen_threshold(self):
        for t, fp, fn in ((None,0,2), (.5,1,1), (1.,0,1)):
            out = evaluate_threshold([1,.25,.5,0], [1,1,0,0], t)
            self.assertEqual((out["false_accepts"], out["false_rejects"]), (fp,fn))
            self.assertEqual(out["balanced_accuracy"], 1-(fp+fn)/4)
            if not fp:
                self.assertAlmostEqual((1-out["zero_false_accept_upper95"])**2,.05)

    def test_canvas(self):
        rng = np.random.default_rng(427)
        sizes = [(1, 100000), (100000, 1), (640, 640), (1,1)]
        sizes.extend(map(tuple,rng.integers(1, 12000, (3000,2)).tolist()))
        for side in (448,640,800):
            for w,h in sizes:
                cells = (side // 32)**2
                scale = min(2*side/max(w,h), max(min(32*c/w, 32*(cells//c)/h)
                            for c in range(1,2*side//32+1)))
                rw,rh = max(1,round(w*scale)), max(1,round(h*scale))
                self.assertEqual(rectangle_shape(w,h,side=side),
                                 (rw,rh,(rw+31)//32*32,(rh+31)//32*32))

    def test_crop(self):
        rng = np.random.default_rng(425)
        for n in (0,1,2,32,64,65,513):
            xy = rng.integers(0, 1000, (n,2))
            faces = [dict(score=.5, bbox=(x-2,y-2,x+2,y+2)) for x,y in xy]
            expected = None
            if n:
                votes = (np.abs(xy[:,None]-xy) <= 250).all(axis=2).sum(axis=1)
                expected = tuple(np.clip(xy[votes.argmax()] - 250,0,500)) + (500,500)
            self.assertEqual(guided_crop(faces,1000,1000),expected)
        self.assertIsNone(guided_crop([dict(score=.1,bbox=(3,5,2,1))],640,640))

    def test_merge(self):
        rng = np.random.default_rng(582)
        for spacing in (12, 1200):
            boxes = rng.integers(0,spacing,(80,2)).tolist()
            faces = [dict(bbox=(x,y,x+12,y+18),score=i%5, keypoints=[(x,y)])
                     for i,(x,y) in enumerate(boxes)]
            base, extra = faces[:40], faces[40:]
            original = copy.deepcopy(extra)
            expected = []
            for i, f in sorted(enumerate(faces),key=lambda p:-p[1]["score"]):
                x,y,r,b = f["bbox"]
                if i >= 40:
                    x,y,r,b = x+3,y+2,r+3,b+2
                if all((lambda area: area <= .4*(432-area))(
                    max(0,min(r,pr)-max(x,px))*max(0,min(b,pb)-max(y,py)))
                       for px,py,pr,pb in expected):
                    expected.append((x,y,r,b))
                if len(expected) == 32:
                    break
            for _ in range(2):
                result = merge_faces(base,extra,3,2)
                self.assertEqual([f["bbox"] for f in result],expected)
                self.assertEqual(extra,original)
                self.assertTrue(all(f["keypoints"][0] == f["bbox"][:2] for f in result))

    def test_sampling(self):
        labels = np.random.default_rng(98).integers(0,120,3000)
        data = np.zeros(len(labels),dtype=INDEX_RECORD)
        data["label"] = labels
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"index.bin"
            data.tofile(path)
            pairs = sample_index_pairs(path,87,123)
            rng = np.random.default_rng(87)
            groups = {}
            for i, label in enumerate(labels):
                groups.setdefault(label,[]).append(i)
            multi = [c for c in groups if len(groups[c]) >= 2]
            ref = [(*rng.choice(groups[c],2,replace=False),True) for c in rng.choice(multi,123)]
            for _ in range(123):
                c,d = rng.choice(multi,2,replace=False)
                ref.append((rng.choice(groups[c]),rng.choice(groups[d]),False))
            rng.shuffle(ref)
            self.assertEqual(pairs,ref)
            path.write_bytes(b"partial")
            with self.assertRaises(ValueError):
                sample_index_pairs(path,87,123)


if __name__ == "__main__":
    unittest.main()
