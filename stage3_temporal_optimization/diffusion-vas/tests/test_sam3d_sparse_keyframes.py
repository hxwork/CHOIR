"""Unit tests for sam3d_sparse_keyframes helpers (no SAM3D inference)."""

import unittest

import torch

from sam3d_sparse_keyframes import (
    _flip_quat_to_match_torch,
    build_rejected_overlay_filename,
    build_sam3d_follow_retry_seeds,
    build_milestone_sampled_indices,
    compress_rotation_follow_indices_non_interaction,
    default_sparse_stride_for_profile,
    sparse_stride_for_motion_profile,
    world_vertices_from_pnp_row,
)


class TestFlipQuat(unittest.TestCase):
    def test_flip_when_opposite_hemisphere(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        ref = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        out = _flip_quat_to_match_torch(q, ref)
        self.assertGreater(float((out.flatten() * ref.flatten()).sum()), 0.0)

    def test_no_flip_when_aligned(self):
        q = torch.tensor([1.0, 0.1, 0.0, 0.0])
        ref = q.clone()
        out = _flip_quat_to_match_torch(q, ref)
        self.assertTrue(torch.allclose(out, q))


class TestMilestones(unittest.TestCase):
    def test_milestone_set_contains_endpoints(self):
        ordered, tags = build_milestone_sampled_indices(
            num_sampled_frames=40,
            stage1_lo_auto=10,
            stage1_hi_auto=30,
            approaching_padded=12,
            interaction_end_padded=28,
            stride_m=8,
        )
        self.assertIn(0, ordered)
        self.assertIn(39, ordered)
        self.assertIn(10, ordered)
        self.assertTrue(any("interaction_every_8" in t for t in tags.values()))


class TestCompressRotationFollow(unittest.TestCase):
    def test_no_bounds_is_identity(self):
        full = [1, 2, 3, 4, 5]
        self.assertEqual(
            compress_rotation_follow_indices_non_interaction(full, interaction_lo=None, interaction_hi=None),
            full,
        )

    def test_all_non_interaction_collapses_to_last_per_block(self):
        out = compress_rotation_follow_indices_non_interaction(
            [1, 2, 3, 10, 11, 12],
            interaction_lo=20,
            interaction_hi=30,
        )
        self.assertEqual(out, [3, 12])

    def test_interaction_keeps_every_index(self):
        out = compress_rotation_follow_indices_non_interaction(
            [21, 22, 23],
            interaction_lo=20,
            interaction_hi=30,
        )
        self.assertEqual(out, [21, 22, 23])


class TestPnpWorldVerts(unittest.TestCase):
    def test_identity_matches_scale_verts(self):
        verts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
        R_col = torch.eye(3)
        t = torch.zeros(3, dtype=torch.float32)
        sc = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        out = world_vertices_from_pnp_row(verts, R_col, t, sc)
        self.assertTrue(torch.allclose(out, verts * sc))


class TestStrideDefaults(unittest.TestCase):
    def test_rotation_dense(self):
        self.assertEqual(default_sparse_stride_for_profile("rotation_likely"), 1)

    def test_translation_sparse(self):
        self.assertEqual(default_sparse_stride_for_profile("translation_likely"), 8)

    def test_override_wins(self):
        self.assertEqual(sparse_stride_for_motion_profile("translation_likely", 3), 3)


class TestRejectedOverlayFilename(unittest.TestCase):
    def test_includes_sample_anchor_and_angle(self):
        self.assertEqual(
            build_rejected_overlay_filename(sampled_idx=13, raw_frame_idx=13, anchor_idx=12, angle_deg=74.1859),
            "overlay_rejected_sidx_0013_raw_00013_anchor_0012_angle_074.2.png",
        )


class TestFollowRetrySeeds(unittest.TestCase):
    def test_zero_retry_keeps_existing_single_attempt_behavior(self):
        self.assertEqual(build_sam3d_follow_retry_seeds(seed=42, retry_count=0), (42,))

    def test_retry_count_adds_extra_random_attempt_seeds(self):
        self.assertEqual(build_sam3d_follow_retry_seeds(seed=42, retry_count=3), (42, 43, 44, 45))


if __name__ == "__main__":
    unittest.main()
