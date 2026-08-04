"""Tests for Stage 5 optimization mode helpers."""

import unittest

import torch

from stage5_mode import (
    STAGE5_FROZEN,
    STAGE5_FULL,
    STAGE5_OBJECT_LITE,
    STAGE5_POSE_ONLY,
    STAGE5_POSE_RAY,
    stage5_hand2d_grad_groups,
    stage5_hand2d_weight,
    stage5_trainable_params,
)
from stage5_object_lite import project_object_translation_delta_
from stage5_ray_delta import build_stage5_ray_delta, clamp_stage5_ray_delta_


class Stage5ModeTest(unittest.TestCase):
    def test_full_mode_trains_object_wrist_and_pose(self):
        self.assertEqual(
            stage5_trainable_params(STAGE5_FULL),
            ("object", "mano_pose", "mano_root"),
        )

    def test_pose_only_mode_trains_only_mano_pose(self):
        self.assertEqual(stage5_trainable_params(STAGE5_POSE_ONLY), ("mano_pose",))

    def test_pose_ray_mode_trains_mano_pose_and_ray_delta(self):
        self.assertEqual(
            stage5_trainable_params(STAGE5_POSE_RAY),
            ("mano_pose", "hand_ray_delta"),
        )

    def test_object_lite_mode_trains_object_and_fingers_only(self):
        self.assertEqual(
            stage5_trainable_params(STAGE5_OBJECT_LITE),
            ("object", "mano_pose"),
        )

    def test_frozen_mode_trains_nothing(self):
        self.assertEqual(stage5_trainable_params(STAGE5_FROZEN), ())

    def test_pose_ray_disables_hand2d(self):
        self.assertEqual(stage5_hand2d_weight(STAGE5_POSE_ONLY), 0.0)
        self.assertEqual(stage5_hand2d_weight(STAGE5_POSE_RAY), 0.0)
        self.assertEqual(stage5_hand2d_weight(STAGE5_OBJECT_LITE), 0.0)
        self.assertEqual(stage5_hand2d_weight(STAGE5_FROZEN), 0.0)
        self.assertEqual(stage5_hand2d_weight(STAGE5_FULL), 5e-1)

    def test_pose_ray_has_no_hand2d_gradients(self):
        self.assertEqual(stage5_hand2d_grad_groups(STAGE5_POSE_RAY), ())
        self.assertEqual(stage5_hand2d_grad_groups(STAGE5_FULL), ("mano_pose", "mano_root"))
        self.assertEqual(stage5_hand2d_grad_groups(STAGE5_POSE_ONLY), ())
        self.assertEqual(stage5_hand2d_grad_groups(STAGE5_OBJECT_LITE), ())
        self.assertEqual(stage5_hand2d_grad_groups(STAGE5_FROZEN), ())

    def test_ray_delta_clamps_and_only_applies_to_interaction(self):
        raw_delta = torch.tensor([-0.05, 0.01, 0.05])
        ray_dirs = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        inter_mask = torch.tensor([False, True, True])

        delta_vec, delta_scalar = build_stage5_ray_delta(
            raw_delta,
            ray_dirs,
            inter_mask,
            min_delta=-0.02,
            max_delta=0.03,
        )

        torch.testing.assert_close(delta_scalar, torch.tensor([0.0, 0.01, 0.03]))
        torch.testing.assert_close(delta_vec[:, 2], torch.tensor([0.0, -0.01, -0.03]))

    def test_ray_delta_inplace_clamp(self):
        raw_delta = torch.nn.Parameter(torch.tensor([-0.05, 0.01, 0.05]))
        clamp_stage5_ray_delta_(raw_delta, min_delta=-0.02, max_delta=0.03)
        torch.testing.assert_close(raw_delta.detach(), torch.tensor([-0.02, 0.01, 0.03]))

    def test_object_lite_translation_projection_limits_delta_norm(self):
        anchor = torch.zeros(3, 3)
        trans = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.03, 0.0, 0.0],
                    [0.0, 0.01, 0.0],
                    [0.0, 0.0, -0.04],
                ]
            )
        )

        project_object_translation_delta_(trans, anchor, max_delta=0.02)

        torch.testing.assert_close(trans.detach()[0], torch.tensor([0.02, 0.0, 0.0]))
        torch.testing.assert_close(trans.detach()[1], torch.tensor([0.0, 0.01, 0.0]))
        torch.testing.assert_close(trans.detach()[2], torch.tensor([0.0, 0.0, -0.02]))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            stage5_trainable_params("unknown")


if __name__ == "__main__":
    unittest.main()
