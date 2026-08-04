"""Tests for adaptive frame sampling."""

import unittest

from frame_sampling import build_adaptive_sample_indices


class FrameSamplingTest(unittest.TestCase):
    def test_short_video_uses_all_frames(self):
        indices, info = build_adaptive_sample_indices(
            30,
            target_stride=3,
            min_sampled_frames=64,
            max_sampled_frames=128,
        )

        self.assertEqual(indices, list(range(30)))
        self.assertEqual(info["num_sampled_frames"], 30)
        self.assertEqual(info["mode"], "all_frames")

    def test_medium_video_keeps_minimum_coverage(self):
        indices, info = build_adaptive_sample_indices(
            90,
            target_stride=3,
            min_sampled_frames=64,
            max_sampled_frames=128,
        )

        self.assertEqual(len(indices), 64)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 89)
        self.assertEqual(info["target_num_sampled_frames"], 64)

    def test_long_video_clamps_to_maximum(self):
        indices, info = build_adaptive_sample_indices(
            600,
            target_stride=3,
            min_sampled_frames=64,
            max_sampled_frames=128,
        )

        self.assertEqual(len(indices), 128)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 599)
        self.assertEqual(info["target_num_sampled_frames"], 128)

    def test_invalid_stride_falls_back_to_one(self):
        indices, info = build_adaptive_sample_indices(
            10,
            target_stride=0,
            min_sampled_frames=64,
            max_sampled_frames=128,
        )

        self.assertEqual(indices, list(range(10)))
        self.assertEqual(info["target_stride"], 1)


if __name__ == "__main__":
    unittest.main()
