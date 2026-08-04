import unittest


from sam3d_reset_policy import (
    choose_reset_bridge_anchor,
    choose_reset_bridge_plan,
    choose_reset_bridge_start,
    should_replace_bridge_frame,
)


class ResetBridgePolicyTest(unittest.TestCase):
    def test_iou_drop_bridge_starts_from_short_margin_before_reset(self):
        start = choose_reset_bridge_start(
            c_frame=36,
            last_anchor_idx=0,
            trigger_reason="iou dropped 0.245 (prev=0.634 cur=0.389)",
            max_bridge_pre_frames=8,
            iou_drop_margin=2,
        )

        self.assertEqual(start, 34)

    def test_consecutive_low_iou_bridge_uses_short_margin(self):
        start = choose_reset_bridge_start(
            c_frame=36,
            last_anchor_idx=0,
            trigger_reason="iou<0.500 for 3 consecutive frames (cur=0.389)",
            max_bridge_pre_frames=8,
            iou_drop_margin=2,
        )

        self.assertEqual(start, 34)

    def test_r_disagreement_bridge_uses_local_window_not_full_anchor_range(self):
        start = choose_reset_bridge_start(
            c_frame=36,
            last_anchor_idx=0,
            trigger_reason="R disagreement 42.0 deg vs SAM3D",
            max_bridge_pre_frames=8,
            iou_drop_margin=2,
        )

        self.assertEqual(start, 28)

    def test_bridge_never_crosses_last_anchor(self):
        start = choose_reset_bridge_start(
            c_frame=5,
            last_anchor_idx=3,
            trigger_reason="R disagreement 42.0 deg vs SAM3D",
            max_bridge_pre_frames=8,
            iou_drop_margin=2,
        )

        self.assertEqual(start, 4)

    def test_choose_nearest_verified_probe_as_bridge_anchor(self):
        sam_cache = {
            22: {"angle_deg": 10.8, "iou_sam": 0.413, "iou_pnp": 0.438},
            30: {"angle_deg": 174.1, "iou_sam": 0.483, "iou_pnp": 0.478},
        }

        anchor = choose_reset_bridge_anchor(
            c_frame=38,
            last_anchor_idx=0,
            sam_cache=sam_cache,
            max_anchor_angle_deg=15.0,
            min_anchor_iou_delta=-0.03,
        )

        self.assertEqual(anchor, 22)

    def test_reject_flipped_probe_as_bridge_anchor(self):
        sam_cache = {
            30: {"angle_deg": 174.1, "iou_sam": 0.483, "iou_pnp": 0.478},
        }

        anchor = choose_reset_bridge_anchor(
            c_frame=38,
            last_anchor_idx=0,
            sam_cache=sam_cache,
            max_anchor_angle_deg=15.0,
            min_anchor_iou_delta=-0.03,
        )

        self.assertEqual(anchor, 0)

    def test_bridge_plan_uses_real_sequential_anchor_but_local_replace_window(self):
        sam_cache = {
            22: {"angle_deg": 10.8, "iou_sam": 0.413, "iou_pnp": 0.438},
            30: {"angle_deg": 174.1, "iou_sam": 0.483, "iou_pnp": 0.478},
        }

        plan = choose_reset_bridge_plan(
            c_frame=38,
            last_anchor_idx=0,
            trigger_reason="R disagreement 45.6 deg vs SAM3D",
            sam_cache=sam_cache,
            max_bridge_pre_frames=8,
            iou_drop_margin=2,
        )

        self.assertEqual(plan["chain_anchor"], 0)
        self.assertEqual(plan["chain_start"], 1)
        self.assertEqual(plan["replace_start"], 23)
        self.assertEqual(plan["trusted_probe_anchor"], 22)

    def test_bridge_plan_uses_validated_stage_static_anchor(self):
        plan = choose_reset_bridge_plan(
            c_frame=38,
            last_anchor_idx=0,
            trigger_reason="R disagreement 45.6 deg vs SAM3D",
            sam_cache={},
            validated_static_anchor_idx=22,
            max_bridge_pre_frames=8,
            iou_drop_margin=2,
        )

        self.assertEqual(plan["chain_anchor"], 22)
        self.assertEqual(plan["chain_start"], 23)
        self.assertEqual(plan["replace_start"], 23)
        self.assertEqual(plan["static_anchor"], 22)

    def test_bridge_frame_replace_allows_small_iou_drop(self):
        self.assertTrue(should_replace_bridge_frame(iou_sam=0.55, iou_pnp=0.57, tolerance=0.03))

    def test_bridge_frame_replace_ignores_iou_drop_by_default(self):
        self.assertTrue(should_replace_bridge_frame(iou_sam=0.09, iou_pnp=0.48, tolerance=0.03))

    def test_bridge_frame_replace_can_use_iou_gate_when_requested(self):
        self.assertFalse(
            should_replace_bridge_frame(
                iou_sam=0.09,
                iou_pnp=0.48,
                tolerance=0.03,
                use_iou_gate=True,
            )
        )

    def test_bridge_frame_replace_rejects_large_rotation_jump(self):
        self.assertFalse(
            should_replace_bridge_frame(
                iou_sam=0.60,
                iou_pnp=0.48,
                angle_to_trusted_deg=173.0,
                max_angle_to_trusted_deg=90.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
