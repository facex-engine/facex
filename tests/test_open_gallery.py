import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"training/scripts"))
from eval_open_gallery import evaluate


class OpenGalleryTest(unittest.TestCase):
    def test_request_and_identity_rates(self):
        a,b,c = np.eye(3,512)
        report = evaluate([a,b],["a","b"],[a,b,c,a,a,c,c,c],
                          ["a","a","b","b","x","y","y","y"],.8,0.)
        self.assertEqual([report[k] for k in ("correct_identifications","known_misidentified",
                                              "known_rejected","fpir","macro_fpir","macro_fnir")],
                         [1,2,1,.25,.5,.75])
        self.assertNotIn("zero_false_id_upper95",report)

    def test_zero_events(self):
        a,b,c = np.eye(3,512)
        r = evaluate([a,b],[1,2],[a,c,c],[1,3,4],.8,0.,True)
        self.assertAlmostEqual((1-r["zero_false_id_upper95"])**2,.05)
        with self.assertRaises(ValueError):
            evaluate([a,b],[1,1],[a,c],[1,2],.8,0.)


if __name__ == "__main__":
    unittest.main()
