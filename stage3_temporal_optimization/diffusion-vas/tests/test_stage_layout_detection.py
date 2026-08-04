"""Tests for automatic stage layout classification."""

import unittest

from stage_layout_detection import (
    LAYOUT_FIVE_STAGE,
    LAYOUT_INTERACTION_ONLY,
    apply_hold_interaction_only_override,
    classify_stage_layout_from_interaction_bounds,
)


class StageLayoutDetectionTest(unittest.TestCase):
    def test_short_prefix_and_suffix_force_interaction_only(self):
        layout = classify_stage_layout_from_interaction_bounds(
            num_sampled_frames=64,
            interaction_start=2,
            interaction_end=62,
            short_side_threshold=5,
        )

        self.assertEqual(layout["layout_mode"], LAYOUT_INTERACTION_ONLY)
        self.assertEqual(layout["pre_interaction_len"], 2)
        self.assertEqual(layout["post_interaction_len"], 2)
        self.assertEqual(layout["reason"], "short_pre_and_post")

    def test_five_stage_when_only_one_side_is_short(self):
        layout = classify_stage_layout_from_interaction_bounds(
            num_sampled_frames=64,
            interaction_start=12,
            interaction_end=62,
            short_side_threshold=5,
        )

        self.assertEqual(layout["layout_mode"], LAYOUT_FIVE_STAGE)
        self.assertEqual(layout["pre_interaction_len"], 12)
        self.assertEqual(layout["post_interaction_len"], 2)

    def test_threshold_is_strictly_less_than(self):
        layout = classify_stage_layout_from_interaction_bounds(
            num_sampled_frames=64,
            interaction_start=5,
            interaction_end=59,
            short_side_threshold=5,
        )

        self.assertEqual(layout["layout_mode"], LAYOUT_FIVE_STAGE)

    def test_hold_video_forces_interaction_only(self):
        layout = classify_stage_layout_from_interaction_bounds(
            num_sampled_frames=30,
            interaction_start=12,
            interaction_end=20,
            short_side_threshold=5,
        )

        overridden = apply_hold_interaction_only_override(
            layout,
            video_id="hold_GPMF14_ho3d",
            num_sampled_frames=30,
        )

        self.assertEqual(overridden["layout_mode"], LAYOUT_INTERACTION_ONLY)
        self.assertEqual(overridden["reason"], "hold_video_forced_interaction_only")
        self.assertEqual(overridden["interaction_start"], 0)
        self.assertEqual(overridden["interaction_end"], 30)
        self.assertEqual(overridden["pre_interaction_len"], 0)
        self.assertEqual(overridden["post_interaction_len"], 0)

    def test_non_hold_video_keeps_detected_layout(self):
        layout = classify_stage_layout_from_interaction_bounds(
            num_sampled_frames=30,
            interaction_start=12,
            interaction_end=20,
            short_side_threshold=5,
        )

        overridden = apply_hold_interaction_only_override(
            layout,
            video_id="IMG_5186",
            num_sampled_frames=30,
        )

        self.assertEqual(overridden, layout)


if __name__ == "__main__":
    unittest.main()
