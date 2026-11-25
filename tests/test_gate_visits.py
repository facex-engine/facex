import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training/scripts"))
from eval_gate_visits import evaluate_visits


def fixture():
    a, b, other = np.eye(3, 512, dtype="f")
    # Visit 3 was logged at the gate but yielded no usable face.
    return dict(enrollment=[a, b], enrolled_ids=["a", "b"], visit_ids=np.arange(1, 8),
                visit_person_ids=["a", "a", "b", "b", "x", "y", "x"],
                probes=np.stack([b, a, a, a, b, b, a, b, a, a, other]),
                frame_visit_ids=np.array([1,1,1,2,4,4,5,5,6,6,7]),
                frame_numbers=np.array([0,1,2,0,0,1,0,1,0,1,0]))


class GateVisitsTest(unittest.TestCase):
    def test_replay_and_input_order(self):
        data = fixture()
        result = evaluate_visits(**data, min_score=.8, min_margin=.05)
        self.assertEqual(result["visits_without_frames"], 1)
        for count, expected in (("1", [2,1,1,2]), ("2", [1,0,2,2])):
            policy = result["consecutive_confirmations"][count]
            keys = ("false_identifications", "known_misidentified", "known_rejected",
                    "correct_identifications")
            self.assertEqual([policy[k] for k in keys], expected)
            self.assertEqual((policy["known_visits"], policy["unknown_visits"]), (4,3))
        order = np.random.default_rng(154).permutation(len(data["probes"]))
        for key in ("probes", "frame_visit_ids", "frame_numbers"):
            data[key] = data[key][order]
        self.assertEqual(evaluate_visits(**data, min_score=.8, min_margin=.05), result)

    def test_gaps_abstentions_and_id_changes(self):
        a, b, zero = np.eye(2,512)[0], np.eye(2,512)[1], np.zeros(512)
        for probes, frames in (([a,a],[1,3]), ([a,zero,a],[1,2,3]), ([a,b],[1,2])):
            data = fixture()
            data.update(probes=probes, frame_visit_ids=[5]*len(probes), frame_numbers=frames)
            policy = evaluate_visits(**data, min_score=.8, min_margin=0)["consecutive_confirmations"]["2"]
            self.assertEqual(policy["false_identifications"],0)
            self.assertEqual(policy["known_rejected"],4)

    def test_invalid_and_empty_logs(self):
        data = fixture()
        for change in (dict(frame_visit_ids=[99]*11), dict(frame_numbers=[0]*11),
                       dict(visit_ids=[1]*7), dict(confirmations=[0])):
            with self.assertRaises(ValueError):
                evaluate_visits(**dict(data, **change), min_score=.8, min_margin=.05)
        data.update(probes=np.empty((0,512)), frame_visit_ids=np.array([],dtype=int),
                    frame_numbers=np.array([],dtype=int))
        report = evaluate_visits(**data, min_score=.8, min_margin=.05)
        self.assertEqual(report["visits_without_frames"],7)
        self.assertEqual(report["consecutive_confirmations"]["2"]["fnir"],1)


if __name__ == "__main__":
    unittest.main()
