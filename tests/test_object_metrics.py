import unittest

import numpy as np

from benchmark.object_metrics import (
    aggregate_scene_metrics,
    make_point_predictions,
    match_instances,
)


class ObjectMetricsTest(unittest.TestCase):
    def test_greedy_matching_counts_duplicates_as_fp(self):
        gt_masks = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
        prediction = {
            "pred_masks": np.array(
                [[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool
            ),
            "pred_scores": np.array([0.9, 0.8, 0.7]),
            "pred_classes": np.array([1, 1, 1]),
        }
        result = match_instances(prediction, gt_masks, np.array([1, 1]))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 0))
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertEqual(result["recall"], 1.0)

    def test_tie_break_prefers_higher_confidence(self):
        gt_masks = np.array([[1, 1, 0]], dtype=bool)
        prediction = {
            "pred_masks": np.array([[1, 1], [1, 1], [0, 0]], dtype=bool),
            "pred_scores": np.array([0.6, 0.9]),
            "pred_classes": np.array([1, 1]),
        }
        result = match_instances(prediction, gt_masks, np.array([1]))
        self.assertEqual(result["matches"][0][0], 1)

    def test_thresholds_are_inclusive_and_classes_must_match(self):
        gt_masks = np.array([[1, 1, 0, 0]], dtype=bool)
        prediction = {
            "pred_masks": np.array(
                [[1, 0], [0, 1], [0, 0], [0, 0]], dtype=bool
            ),
            "pred_scores": np.array([0.5, 1.0]),
            "pred_classes": np.array([1, 0]),
        }
        result = match_instances(prediction, gt_masks, np.array([1]))
        # First prediction has exactly score=.5 and IoU=.5. Class-0 is ignored.
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_zero_denominators_and_global_accumulation(self):
        empty = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "per_class": {1: {"tp": 0, "fp": 0, "fn": 0}},
        }
        total = aggregate_scene_metrics({"a": empty})
        self.assertEqual(total["precision"], 0.0)
        self.assertEqual(total["recall"], 0.0)
        self.assertEqual(total["f1"], 0.0)

    def test_las_overlap_uses_highest_score_and_contiguous_ids(self):
        prediction = {
            "pred_masks": np.array([[1, 1], [1, 0], [0, 1]], dtype=bool),
            "pred_scores": np.array([0.5, 0.9]),
            "pred_classes": np.array([1, 1]),
        }
        semantic, instance, score = make_point_predictions(prediction, 3)
        np.testing.assert_array_equal(semantic, [1, 1, 1])
        np.testing.assert_array_equal(instance, [1, 2, 1])
        np.testing.assert_allclose(score, [0.9, 0.5, 0.9])


if __name__ == "__main__":
    unittest.main()
