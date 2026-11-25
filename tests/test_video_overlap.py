"""Compare the video cache overlap index with exhaustive pairwise checks."""
import random
import unittest

from facex.video import _ambiguous_tracks, _iou


class VideoOverlapTest(unittest.TestCase):
    def check_boxes(self, boxes, marked=()):
        tracks = [{"ambiguous": i in marked} for i in range(len(boxes))]
        expected = [track["ambiguous"] or any(
            j != i and _iou(box, other) > .15 for j, other in enumerate(boxes)
        ) for i, (box, track) in enumerate(zip(boxes, tracks))]
        self.assertEqual(_ambiguous_tracks(boxes, tracks), expected)

    def test_overlap_boundaries(self):
        # Two widths of 23 with intersection 6 have IoU exactly 0.15.
        for offset in (17, 16.999999, 17.000001, 23, 0):
            self.check_boxes([(0, 0, 23, 8), (offset, 0, offset + 23, 8)])
        self.check_boxes([(0, 0, 100, 10), (10, 10, 15, 20), (78, 0, 99, 10)], {0})

    def test_random_layouts(self):
        rng = random.Random(128)
        for n in (0, 1, 2, 8, 32, 128):
            for extent in (8, 100, 1000):
                boxes = []
                for _ in range(n):
                    x, y = rng.uniform(-extent, extent), rng.uniform(-extent, extent)
                    boxes.append((x, y, x + rng.uniform(1, 100), y + rng.uniform(1, 100)))
                self.check_boxes(boxes, range(0, n, 9))


if __name__ == "__main__":
    unittest.main()
