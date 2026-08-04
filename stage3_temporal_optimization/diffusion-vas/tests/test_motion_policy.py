"""Tests for global motion-policy overrides."""

import unittest

from motion_policy import FORCE_ROTATION_MODE, force_rotation_likely_profile


class MotionPolicyTest(unittest.TestCase):
    def test_force_rotation_preserves_original_mode_for_debug(self):
        profile = {"suggested_mode": "translation_likely", "score": 1.0}

        out = force_rotation_likely_profile(profile, reason="unit_test")

        self.assertEqual(out["suggested_mode"], FORCE_ROTATION_MODE)
        self.assertEqual(out["policy_original_suggested_mode"], "translation_likely")
        self.assertEqual(out["policy_override"], "force_rotation_likely")
        self.assertEqual(out["policy_override_reason"], "unit_test")
        self.assertEqual(out["score"], 1.0)

    def test_force_rotation_handles_missing_profile(self):
        out = force_rotation_likely_profile(None, reason="unit_test")

        self.assertEqual(out["suggested_mode"], FORCE_ROTATION_MODE)
        self.assertIsNone(out["policy_original_suggested_mode"])


if __name__ == "__main__":
    unittest.main()
