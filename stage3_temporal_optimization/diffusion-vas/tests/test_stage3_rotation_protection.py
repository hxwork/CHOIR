"""Tests for Stage 3 SAM3D rotation protection helpers."""

import math
import unittest

import torch
from pytorch3d.transforms import axis_angle_to_matrix

from stage3_rotation_protection import (
    rotation_drift_degrees,
    stage3_rotation_lr_for_step,
    stage3_rotation_smoothness_scale,
)


class Stage3RotationProtectionTest(unittest.TestCase):
    def test_rotation_lr_freezes_then_unlocks(self):
        self.assertEqual(stage3_rotation_lr_for_step(0, 100, 1e-2), 0.0)
        self.assertEqual(stage3_rotation_lr_for_step(64, 100, 1e-2), 0.0)
        self.assertAlmostEqual(stage3_rotation_lr_for_step(65, 100, 1e-2), 5e-4)
        self.assertAlmostEqual(stage3_rotation_lr_for_step(99, 100, 1e-2), 5e-4)

    def test_rotation_lr_direct_optimization_uses_base_lr_immediately(self):
        self.assertAlmostEqual(
            stage3_rotation_lr_for_step(0, 100, 1e-2, protect_rotation=False),
            1e-2,
        )
        self.assertAlmostEqual(
            stage3_rotation_lr_for_step(64, 100, 1e-2, protect_rotation=False),
            1e-2,
        )
        self.assertAlmostEqual(
            stage3_rotation_lr_for_step(99, 100, 1e-2, protect_rotation=False),
            1e-2,
        )

    def test_rotation_smoothness_scale_is_light_by_default(self):
        self.assertAlmostEqual(stage3_rotation_smoothness_scale(), 0.05)
        self.assertAlmostEqual(stage3_rotation_smoothness_scale(0.1), 0.1)

    def test_rotation_drift_degrees_reports_geodesic_angles(self):
        anchor = torch.eye(3).view(1, 3, 3).repeat(2, 1, 1)
        current = torch.stack(
            [
                axis_angle_to_matrix(torch.tensor([0.0, 0.0, 0.0])),
                axis_angle_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2])),
            ],
            dim=0,
        )

        stats = rotation_drift_degrees(current, anchor)

        self.assertAlmostEqual(stats["mean_deg"], 45.0, places=3)
        self.assertAlmostEqual(stats["max_deg"], 90.0, places=3)


if __name__ == "__main__":
    unittest.main()
