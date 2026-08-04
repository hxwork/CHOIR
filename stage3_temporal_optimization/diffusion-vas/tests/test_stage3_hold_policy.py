"""Tests for hold-only Stage 3 hand fitting policy."""

import unittest

import numpy as np

from stage3_hold_policy import (
    hold_stage3_hand_refine_steps,
    hold_joint_target_indices,
    hold_mano_init_param_groups,
    hold_mano_init_path,
    hold_pose_from_fit,
    hold_pose_hand_mean,
    hold_reorder_valid_mask,
    hold_sam3d_rot_outlier_max_angle,
    is_hold_video_id,
    stage3_object_smoothness_scale,
    stage3_joint2d_relative_weight,
)


class Stage3HoldPolicyTest(unittest.TestCase):
    def test_hold_detection_only_matches_hold_prefix(self):
        self.assertTrue(is_hold_video_id("hold_ABF12_ho3d"))
        self.assertFalse(is_hold_video_id("84319"))
        self.assertFalse(is_hold_video_id("IMG_5142"))

    def test_hold_uses_stronger_joint2d_weight(self):
        self.assertEqual(stage3_joint2d_relative_weight("hold_ABF12_ho3d"), 5.0)
        self.assertEqual(stage3_joint2d_relative_weight("84319"), 1.0)

    def test_hold_uses_default_sam3d_rotation_outlier_threshold(self):
        self.assertEqual(hold_sam3d_rot_outlier_max_angle("hold_GPMF12_ho3d", default=60.0), 60.0)
        self.assertEqual(hold_sam3d_rot_outlier_max_angle("84319", default=60.0), 60.0)

    def test_hold_enables_hand_only_refinement(self):
        self.assertEqual(hold_stage3_hand_refine_steps("hold_ABF12_ho3d"), 200)
        self.assertEqual(hold_stage3_hand_refine_steps("84319"), 0)

    def test_hold_disables_stage3_object_smoothness(self):
        self.assertEqual(stage3_object_smoothness_scale("hold_ABF12_ho3d"), 0.0)
        self.assertEqual(stage3_object_smoothness_scale("84319"), 1.0)

    def test_hold_joint_target_order_matches_openpose_mapping(self):
        self.assertEqual(
            hold_joint_target_indices(),
            (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20),
        )

    def test_hold_mano_init_path(self):
        self.assertEqual(
            hold_mano_init_path("/data/hold_ABF12_ho3d"),
            "/data/hold_ABF12_ho3d/processed/hold_fit.slerp.npy",
        )

    def test_hold_mano_init_only_overrides_finger_pose(self):
        self.assertEqual(hold_mano_init_param_groups(), ("mano_pose",))

    def test_hold_pose_hand_mean_accepts_flat_mano_mean(self):
        flat_mean = np.arange(45, dtype=np.float32)

        hand_mean = hold_pose_hand_mean(flat_mean)

        self.assertEqual(hand_mean.shape, (15, 3))
        np.testing.assert_array_equal(hand_mean.reshape(-1), flat_mean)

    def test_hold_pose_from_fit_keeps_temporal_model_shape(self):
        hand_pose = np.zeros((2, 45), dtype=np.float32)
        hand_mean = np.arange(45, dtype=np.float32).reshape(15, 3)

        pose = hold_pose_from_fit(hand_pose, hand_mean)

        self.assertEqual(pose.shape, (2, 15, 3))
        np.testing.assert_array_equal(pose[0], hand_mean)

    def test_hold_reorder_valid_mask_keeps_frame_level_mask(self):
        valid_mask = np.array([True, False, True])

        reordered = hold_reorder_valid_mask(valid_mask, (2, 1, 0))

        self.assertIs(reordered, valid_mask)

    def test_hold_reorder_valid_mask_reorders_joint_level_mask(self):
        valid_mask = np.array([[True, False, False], [False, True, True]])

        reordered = hold_reorder_valid_mask(valid_mask, (2, 1, 0))

        np.testing.assert_array_equal(
            reordered,
            np.array([[False, False, True], [True, True, False]]),
        )


if __name__ == "__main__":
    unittest.main()
