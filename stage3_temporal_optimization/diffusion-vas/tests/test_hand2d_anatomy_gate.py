"""Tests for anatomy-driven hand 2D frame weighting."""

import unittest

import torch

from hand2d_anatomy_gate import compute_anatomy_hand2d_frame_weights


class Hand2DAnatomyGateTest(unittest.TestCase):
    def test_extreme_anatomy_outliers_are_dropped(self):
        scores = torch.tensor([0.10, 0.11, 0.09, 1.20, 0.10], dtype=torch.float32)

        weights, info = compute_anatomy_hand2d_frame_weights(scores)

        self.assertEqual(float(weights[3]), 0.0)
        self.assertTrue(torch.allclose(weights[[0, 1, 2, 4]], torch.ones(4)))
        self.assertEqual(info["n_low"], 1)
        self.assertIn(3, info["low_indices"])
        self.assertIn(3, info["extreme_indices"])

    def test_moderate_anatomy_variation_keeps_full_weight(self):
        scores = torch.tensor([0.10, 0.11, 0.09, 0.16, 0.10], dtype=torch.float32)

        weights, info = compute_anatomy_hand2d_frame_weights(scores)

        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))
        self.assertEqual(info["n_low"], 0)
        self.assertEqual(info["extreme_indices"], [])

    def test_uniform_scores_keep_full_weight(self):
        scores = torch.full((6,), 0.2, dtype=torch.float32)

        weights, info = compute_anatomy_hand2d_frame_weights(scores)

        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))
        self.assertEqual(info["n_low"], 0)


if __name__ == "__main__":
    unittest.main()
