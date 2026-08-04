"""Tests for applying GFM camera-ray depth offsets."""

import unittest

import torch

from gfm_offset_application import apply_camera_ray_depth_offsets


class GfmOffsetApplicationTest(unittest.TestCase):
    def test_matches_original_frame_key_before_sample_position_key(self):
        mano_trans = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        sampled_indices = [0, 10]
        offsets = {
            "0": 0.1,
            "1": 999.0,
            "10": 0.2,
        }

        corrected, stats = apply_camera_ray_depth_offsets(
            mano_trans,
            sampled_indices,
            offsets,
        )

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.9],
                [0.0, 0.0, 1.8],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(corrected, expected)
        self.assertEqual(stats["matched_by_frame"], 2)
        self.assertEqual(stats["matched_by_position"], 0)
        self.assertEqual(stats["applied_count"], 2)

    def test_falls_back_to_sample_position_for_legacy_offsets(self):
        mano_trans = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)

        corrected, stats = apply_camera_ray_depth_offsets(
            mano_trans,
            [10],
            {"0": 0.25},
        )

        torch.testing.assert_close(corrected, torch.tensor([[0.0, 0.0, 1.75]]))
        self.assertEqual(stats["matched_by_frame"], 0)
        self.assertEqual(stats["matched_by_position"], 1)

    def test_smoothing_only_changes_offsets_inside_requested_range(self):
        mano_trans = torch.tensor([[0.0, 0.0, 2.0]] * 5, dtype=torch.float32)
        offsets = {
            "0": 0.10,
            "1": 0.00,
            "2": 0.09,
            "3": 0.00,
            "4": 0.20,
        }

        corrected, stats = apply_camera_ray_depth_offsets(
            mano_trans,
            [0, 1, 2, 3, 4],
            offsets,
            smooth_offsets=True,
            smooth_range=(1, 4),
            smooth_kernel=(1.0, 2.0, 1.0),
        )

        applied = 2.0 - corrected[:, 2]
        torch.testing.assert_close(applied, torch.tensor([0.10, 0.03, 0.045, 0.03, 0.20]))
        self.assertTrue(stats["smoothed"])
        self.assertEqual(stats["smooth_range"], [1, 4])
        self.assertGreater(stats["mean_abs_smoothing_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
