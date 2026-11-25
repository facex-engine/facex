"""Check canvas bytes against the original Pillow preparation path."""
import unittest

import numpy as np
from PIL import Image

from facex.yunet import prepare_rectangle, rectangle_shape


class RectanglePixelsTest(unittest.TestCase):
    def test_resize_padding_and_channel_order(self):
        rng = np.random.default_rng(41)
        for w, h in ((640, 640), (832, 468), (451, 797), (643, 480), (1, 127)):
            rgb = rng.integers(256, size=(h, w, 3), dtype=np.uint8)[:, ::-1]
            for side in (448, 640, 800):
                rw, rh, cw, ch = rectangle_shape(w, h, side=side)
                resized = np.asarray(Image.fromarray(rgb).resize(
                    (rw, rh), getattr(Image, "Resampling", Image).BICUBIC))
                for bgr in (False, True):
                    expected = np.zeros((ch, cw, 3), dtype=np.uint8)
                    x, y = (cw - rw) // 2, (ch - rh) // 2
                    expected[y:y+rh, x:x+rw] = resized[:, :, ::-1] if bgr else resized
                    actual, aw, ah = prepare_rectangle(rgb, side=side, bgr=bgr)
                    self.assertEqual((rw, rh), (aw, ah))
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(actual.flags.c_contiguous and actual.flags.writeable)
                    self.assertFalse(np.shares_memory(rgb, actual))


if __name__ == "__main__":
    unittest.main()
