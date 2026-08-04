import unittest

import torch

from contact_constraint import (
    ContactCache,
    compute_confident_gap_closing_loss,
    compute_contact_patch_centroid_loss,
    compute_interaction_alignment_offsets,
    compute_grasp_template_anchor_loss,
    compute_interaction_boundary_blend_offsets,
    compute_neighbor_template_bridge_loss,
    compute_soft_contact_loss,
    expand_reliable_contact_frontier,
    propagate_grasp_template_from_reliable_frames,
    propagate_dense_contact_memory,
    temporal_lock_argmax,
)


class ContactConfidenceTest(unittest.TestCase):
    def _cache(self, contact_confidence):
        return ContactCache(
            bucket=torch.tensor([[2, 2]], dtype=torch.int8),
            signed_dist=torch.zeros(1, 2),
            face_id_topk=torch.tensor([[[0], [0]]], dtype=torch.long),
            bary_topk=torch.tensor([[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]], dtype=torch.float32),
            weight_topk=torch.ones(1, 2, 1),
            active_idx=torch.tensor([[0, 1]], dtype=torch.long),
            active_mask=torch.tensor([[True, True]]),
            argmax_face=torch.tensor([[0, 0]], dtype=torch.long),
            contact_confidence=contact_confidence,
        )

    def test_contact_confidence_downweights_loss(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        full_loss, full_info = compute_soft_contact_loss(
            hand, obj, faces, self._cache(torch.ones(1, 2)))
        weighted_loss, weighted_info = compute_soft_contact_loss(
            hand, obj, faces, self._cache(torch.tensor([[1.0, 0.0]])))

        self.assertAlmostEqual(float(full_loss), 2.5, places=5)
        self.assertAlmostEqual(float(weighted_loss), 1.0, places=5)
        self.assertEqual(full_info["n_active"], 2)
        self.assertEqual(weighted_info["n_active"], 1)

    def test_temporal_lock_preserves_contact_confidence(self):
        confidence = torch.tensor([[0.25, 0.75]])
        cache = self._cache(confidence)
        cache.observed_contact_confidence = torch.tensor([[0.1, 0.2]])
        cache.raw_observed_contact_confidence = torch.tensor([[0.0, 0.1]])
        cache.raw_geometry_contact_confidence = torch.tensor([[0.3, 0.4]])

        locked = temporal_lock_argmax(cache, window=0, min_consensus=1)

        self.assertTrue(torch.equal(locked.contact_confidence, confidence))
        self.assertTrue(torch.equal(locked.observed_contact_confidence, torch.tensor([[0.1, 0.2]])))
        self.assertTrue(torch.equal(locked.raw_observed_contact_confidence, torch.tensor([[0.0, 0.1]])))
        self.assertTrue(torch.equal(locked.raw_geometry_contact_confidence, torch.tensor([[0.3, 0.4]])))

    def test_dense_contact_memory_propagates_across_frames(self):
        confidence = torch.zeros(5, 2)
        confidence[1, 0] = 0.8
        confidence[2, 0] = 0.9
        anchor_face = torch.full((5, 2), -1, dtype=torch.long)
        anchor_face[1:3, 0] = 7
        anchor_bary = torch.zeros(5, 2, 3)
        anchor_bary[1:3, 0] = torch.tensor([0.2, 0.3, 0.5])

        out_conf, out_face, out_bary = propagate_dense_contact_memory(
            confidence,
            anchor_face,
            anchor_bary,
            temporal_decay=0.5,
            max_steps=2,
            min_seed_conf=0.5,
            min_keep_conf=0.05,
        )

        self.assertGreater(float(out_conf[0, 0]), 0.0)
        self.assertGreater(float(out_conf[3, 0]), 0.0)
        self.assertEqual(int(out_face[3, 0]), 7)
        self.assertTrue(torch.allclose(out_bary[3, 0], torch.tensor([0.2, 0.3, 0.5])))
        self.assertEqual(float(out_conf[4, 1]), 0.0)

    def test_dense_contact_memory_release_gate_blocks_propagation(self):
        confidence = torch.zeros(4, 1)
        confidence[1, 0] = 0.9
        anchor_face = torch.full((4, 1), -1, dtype=torch.long)
        anchor_face[1, 0] = 3
        anchor_bary = torch.zeros(4, 1, 3)
        anchor_bary[1, 0] = torch.tensor([0.1, 0.2, 0.7])
        release_mask = torch.zeros(4, 1, dtype=torch.bool)
        release_mask[2, 0] = True

        out_conf, out_face, _ = propagate_dense_contact_memory(
            confidence,
            anchor_face,
            anchor_bary,
            temporal_decay=0.75,
            max_steps=1,
            release_mask=release_mask,
        )

        self.assertEqual(float(out_conf[2, 0]), 0.0)
        self.assertEqual(int(out_face[2, 0]), -1)

    def test_grasp_template_propagates_from_reliable_frames(self):
        confidence = torch.zeros(6, 3)
        confidence[1, :2] = torch.tensor([0.8, 0.7])
        confidence[2, :2] = torch.tensor([0.9, 0.85])
        anchor_face = torch.full((6, 3), -1, dtype=torch.long)
        anchor_face[1:3, :2] = torch.tensor([[4, 5], [4, 5]])
        anchor_bary = torch.zeros(6, 3, 3)
        anchor_bary[1:3, 0] = torch.tensor([0.2, 0.3, 0.5])
        anchor_bary[1:3, 1] = torch.tensor([0.1, 0.7, 0.2])

        out_conf, out_face, out_bary, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.6,
            min_seed_contacts=2,
            temporal_decay=0.7,
            max_steps=3,
        )

        self.assertTrue(torch.equal(reliable, torch.tensor([False, True, True, False, False, False])))
        self.assertGreater(float(out_conf[4, 0]), 0.0)
        self.assertEqual(int(out_face[4, 0]), 4)
        self.assertTrue(torch.allclose(out_bary[4, 1], torch.tensor([0.1, 0.7, 0.2])))

    def test_grasp_template_requires_reliable_frame_contact_count(self):
        confidence = torch.zeros(4, 2)
        confidence[1, 0] = 0.9
        anchor_face = torch.full((4, 2), -1, dtype=torch.long)
        anchor_face[1, 0] = 2
        anchor_bary = torch.zeros(4, 2, 3)
        anchor_bary[1, 0] = torch.tensor([0.3, 0.3, 0.4])

        out_conf, out_face, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.6,
            min_seed_contacts=2,
            temporal_decay=0.7,
            max_steps=2,
        )

        self.assertFalse(reliable.any())
        self.assertEqual(float(out_conf[2, 0]), 0.0)
        self.assertEqual(int(out_face[2, 0]), -1)

    def test_grasp_template_seed_can_be_blocked_by_frame_mask(self):
        confidence = torch.zeros(5, 2)
        confidence[1:4, :] = 0.9
        anchor_face = torch.full((5, 2), -1, dtype=torch.long)
        anchor_face[1:4, :] = 2
        anchor_bary = torch.zeros(5, 2, 3)
        anchor_bary[1:4, :] = torch.tensor([0.3, 0.3, 0.4])
        close_seed_frames = torch.tensor([False, False, False, False, False])

        out_conf, out_face, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.6,
            min_seed_contacts=2,
            min_reliable_run=3,
            temporal_decay=0.7,
            max_steps=2,
            seed_frame_mask=close_seed_frames,
        )

        self.assertFalse(reliable.any())
        self.assertEqual(float(out_conf.sum()), 0.0)
        self.assertTrue(torch.equal(out_face, torch.full_like(out_face, -1)))

    def test_grasp_template_reliable_seed_uses_observed_confidence_only(self):
        confidence = torch.zeros(4, 2)
        confidence[1, :] = torch.tensor([0.9, 0.9])
        confidence[2, :] = torch.tensor([0.9, 0.9])
        observed_confidence = torch.zeros(4, 2)
        observed_confidence[1, :] = torch.tensor([0.9, 0.9])
        anchor_face = torch.full((4, 2), -1, dtype=torch.long)
        anchor_face[1:3, :] = 4
        anchor_bary = torch.zeros(4, 2, 3)
        anchor_bary[1:3, :] = torch.tensor([0.2, 0.3, 0.5])

        _, out_face, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.6,
            min_seed_contacts=2,
            temporal_decay=0.5,
            max_steps=1,
            seed_confidence=observed_confidence,
        )

        self.assertTrue(torch.equal(reliable, torch.tensor([False, True, False, False])))
        self.assertEqual(int(out_face[2, 0]), 4)

    def test_grasp_template_keeps_only_longest_contiguous_reliable_core(self):
        confidence = torch.zeros(10, 2)
        confidence[[0, 6, 7, 8, 9], :] = 0.9
        anchor_face = torch.full((10, 2), -1, dtype=torch.long)
        anchor_face[[0, 6, 7, 8, 9], :] = 4
        anchor_bary = torch.zeros(10, 2, 3)
        anchor_bary[[0, 6, 7, 8, 9], :] = torch.tensor([0.2, 0.3, 0.5])

        out_conf, out_face, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.6,
            min_seed_contacts=2,
            min_reliable_run=3,
            temporal_decay=0.5,
            max_steps=1,
        )

        expected = torch.tensor([False, False, False, False, False, False, True, True, True, True])
        self.assertTrue(torch.equal(reliable, expected))
        self.assertEqual(float(out_conf[0, 0]), 0.0)
        self.assertEqual(int(out_face[0, 0]), -1)
        self.assertGreater(float(out_conf[5, 0]), 0.0)

    def test_grasp_template_expands_core_only_to_consistent_neighbors(self):
        confidence = torch.zeros(8, 4)
        # Core frames: strong contiguous reliable contact on verts 0,1.
        confidence[3:5, 0:2] = 0.9
        # Neighbor candidates: lower confidence, enough contacts to be checked
        # by expansion but not enough to become core seeds directly.
        confidence[2, 0:2] = 0.55
        confidence[5, 0:2] = 0.55
        confidence[1, 0:2] = 0.55
        confidence[6, 0:2] = 0.55

        anchor_face = torch.full((8, 4), -1, dtype=torch.long)
        anchor_face[3:5, 0] = 10
        anchor_face[3:5, 1] = 11
        # These two agree with the current template and should be included.
        anchor_face[2, 0] = 10
        anchor_face[2, 1] = 11
        anchor_face[5, 0] = 10
        anchor_face[5, 1] = 11
        # These have enough observed contacts but disagree, so expansion stops.
        anchor_face[1, 0] = 20
        anchor_face[1, 1] = 21
        anchor_face[6, 0] = 30
        anchor_face[6, 1] = 31
        anchor_bary = torch.zeros(8, 4, 3)
        anchor_bary[anchor_face >= 0] = torch.tensor([0.2, 0.3, 0.5])

        _, _, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.8,
            min_seed_contacts=2,
            min_reliable_run=2,
            expand_reliable=True,
            min_expand_conf=0.5,
            min_expand_contacts=2,
            min_expand_overlap=0.5,
            min_expand_face_agree=0.75,
            temporal_decay=0.5,
            max_steps=1,
        )

        expected = torch.tensor([False, False, True, True, True, True, False, False])
        self.assertTrue(torch.equal(reliable, expected))

    def test_grasp_template_bootstraps_weak_contiguous_core_when_strict_core_absent(self):
        confidence = torch.zeros(6, 4)
        confidence[2:4, 0:2] = 0.4
        anchor_face = torch.full((6, 4), -1, dtype=torch.long)
        anchor_face[2:4, 0] = 10
        anchor_face[2:4, 1] = 11
        anchor_bary = torch.zeros(6, 4, 3)
        anchor_bary[2:4, 0:2] = torch.tensor([0.2, 0.3, 0.5])

        out_conf, out_face, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.45,
            min_seed_contacts=3,
            min_reliable_run=3,
            bootstrap_if_no_core=True,
            bootstrap_min_seed_conf=0.35,
            bootstrap_min_seed_contacts=2,
            bootstrap_min_reliable_run=2,
            temporal_decay=0.5,
            max_steps=1,
        )

        expected = torch.tensor([False, False, True, True, False, False])
        self.assertTrue(torch.equal(reliable, expected))
        self.assertGreater(float(out_conf[2, 0]), 0.0)
        self.assertEqual(int(out_face[3, 1]), 11)

    def test_grasp_template_prefers_strict_core_over_bootstrap_candidates(self):
        confidence = torch.zeros(8, 4)
        confidence[1:3, 0:2] = 0.4
        confidence[4:7, 0:3] = 0.9
        anchor_face = torch.full((8, 4), -1, dtype=torch.long)
        anchor_face[1:3, 0:2] = 20
        anchor_face[4:7, 0:3] = 10
        anchor_bary = torch.zeros(8, 4, 3)
        anchor_bary[anchor_face >= 0] = torch.tensor([0.2, 0.3, 0.5])

        _, _, _, reliable = propagate_grasp_template_from_reliable_frames(
            confidence,
            anchor_face,
            anchor_bary,
            min_seed_conf=0.45,
            min_seed_contacts=3,
            min_reliable_run=3,
            bootstrap_if_no_core=True,
            bootstrap_min_seed_conf=0.35,
            bootstrap_min_seed_contacts=2,
            bootstrap_min_reliable_run=2,
            temporal_decay=0.5,
            max_steps=1,
        )

        expected = torch.tensor([False, False, False, False, True, True, True, False])
        self.assertTrue(torch.equal(reliable, expected))

    def test_reliable_contact_frontier_expands_only_consistent_neighbors(self):
        observed_confidence = torch.zeros(8, 4)
        observed_confidence[1, 0:2] = 0.6
        observed_confidence[6, 0:2] = 0.6
        observed_confidence[0, 0:2] = 0.6
        observed_confidence[7, 0:2] = 0.6
        observed_face = torch.full((8, 4), -1, dtype=torch.long)
        observed_face[1, 0] = 10
        observed_face[1, 1] = 11
        observed_face[6, 0] = 10
        observed_face[6, 1] = 11
        observed_face[0, 0] = 20
        observed_face[0, 1] = 21
        observed_face[7, 0] = 30
        observed_face[7, 1] = 31
        template_face = torch.full((8, 4), -1, dtype=torch.long)
        template_face[1:7, 0] = 10
        template_face[1:7, 1] = 11
        reliable = torch.tensor([False, False, True, True, True, True, False, False])

        expanded = expand_reliable_contact_frontier(
            observed_confidence,
            observed_face,
            template_face,
            reliable,
            min_expand_conf=0.5,
            min_expand_contacts=2,
            min_expand_overlap=0.5,
            min_expand_face_agree=0.75,
            max_new_frames_per_side=1,
        )

        expected = torch.tensor([False, True, True, True, True, True, True, False])
        self.assertTrue(torch.equal(expanded, expected))

    def test_reliable_contact_frontier_expands_multiple_frames_without_jumping_failures(self):
        observed_confidence = torch.zeros(10, 4)
        observed_confidence[0:3, 0:2] = 0.6
        observed_confidence[7:10, 0:2] = 0.6
        observed_face = torch.full((10, 4), -1, dtype=torch.long)
        observed_face[1:3, 0] = 10
        observed_face[1:3, 1] = 11
        observed_face[7:9, 0] = 10
        observed_face[7:9, 1] = 11
        # These outer frames have enough contacts but disagree with the template.
        observed_face[0, 0] = 20
        observed_face[0, 1] = 21
        observed_face[9, 0] = 30
        observed_face[9, 1] = 31
        template_face = torch.full((10, 4), -1, dtype=torch.long)
        template_face[:, 0] = 10
        template_face[:, 1] = 11
        reliable = torch.tensor([False, False, False, True, True, True, True, False, False, False])

        expanded = expand_reliable_contact_frontier(
            observed_confidence,
            observed_face,
            template_face,
            reliable,
            min_expand_conf=0.5,
            min_expand_contacts=2,
            min_expand_overlap=0.5,
            min_expand_face_agree=0.75,
            max_new_frames_per_side=3,
        )

        expected = torch.tensor([False, True, True, True, True, True, True, True, True, False])
        self.assertTrue(torch.equal(expanded, expected))

    def test_reliable_contact_frontier_unlimited_expands_until_first_failure(self):
        observed_confidence = torch.zeros(12, 4)
        observed_confidence[0:4, 0:2] = 0.6
        observed_confidence[8:12, 0:2] = 0.6
        observed_face = torch.full((12, 4), -1, dtype=torch.long)
        observed_face[1:4, 0] = 10
        observed_face[1:4, 1] = 11
        observed_face[8:11, 0] = 10
        observed_face[8:11, 1] = 11
        observed_face[0, 0] = 20
        observed_face[0, 1] = 21
        observed_face[11, 0] = 30
        observed_face[11, 1] = 31
        template_face = torch.full((12, 4), -1, dtype=torch.long)
        template_face[:, 0] = 10
        template_face[:, 1] = 11
        reliable = torch.tensor([False, False, False, False, True, True, True, True, False, False, False, False])

        expanded = expand_reliable_contact_frontier(
            observed_confidence,
            observed_face,
            template_face,
            reliable,
            min_expand_conf=0.5,
            min_expand_contacts=2,
            min_expand_overlap=0.5,
            min_expand_face_agree=0.75,
            max_new_frames_per_side=-1,
        )

        expected = torch.tensor([False, True, True, True, True, True, True, True, True, True, True, False])
        self.assertTrue(torch.equal(expanded, expected))

    def test_boundary_blend_offsets_ramp_to_jump_at_interaction_end(self):
        offsets, info = compute_interaction_boundary_blend_offsets(
            num_frames=6,
            i0=1,
            i1=4,
            jump=torch.tensor([0.03, 0.0, 0.0]),
            gamma=1.0,
            max_jump=0.05,
        )

        self.assertTrue(torch.allclose(offsets[0], torch.zeros(3)))
        self.assertTrue(torch.allclose(offsets[1], torch.zeros(3)))
        self.assertTrue(torch.allclose(offsets[2], torch.tensor([0.015, 0.0, 0.0])))
        self.assertTrue(torch.allclose(offsets[3], torch.tensor([0.03, 0.0, 0.0])))
        self.assertTrue(torch.allclose(offsets[4], torch.zeros(3)))
        self.assertAlmostEqual(info["jump_norm"], 0.03, places=6)
        self.assertAlmostEqual(info["applied_max_norm"], 0.03, places=6)

    def test_boundary_blend_offsets_clamps_large_jump(self):
        offsets, info = compute_interaction_boundary_blend_offsets(
            num_frames=3,
            i0=0,
            i1=2,
            jump=torch.tensor([0.10, 0.0, 0.0]),
            gamma=1.0,
            max_jump=0.04,
        )

        self.assertTrue(torch.allclose(offsets[1], torch.tensor([0.04, 0.0, 0.0])))
        self.assertAlmostEqual(info["jump_norm"], 0.10, places=6)
        self.assertAlmostEqual(info["applied_max_norm"], 0.04, places=6)

    def test_interaction_alignment_offsets_keep_object_and_hand_boundaries_continuous(self):
        obj_trans = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        hand_trans = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 2.0, 0.0],
            [0.4, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ], dtype=torch.float32)

        obj_offsets, hand_offsets, info = compute_interaction_alignment_offsets(
            obj_trans,
            hand_trans,
            i0=2,
            i1=4,
            pre_buffer=2,
            post_buffer=2,
            max_offset=2.0,
            gamma=1.0,
        )

        corrected_obj = obj_trans + obj_offsets
        corrected_hand = hand_trans + hand_offsets
        self.assertTrue(torch.allclose(corrected_obj[2], obj_trans[1]))
        self.assertTrue(torch.allclose(corrected_obj[3], obj_trans[4]))
        self.assertTrue(torch.allclose(corrected_hand[1], corrected_hand[2]))
        self.assertTrue(torch.allclose(corrected_hand[3], corrected_hand[4]))
        self.assertTrue(torch.allclose(obj_offsets[0], torch.zeros(3)))
        self.assertTrue(torch.allclose(obj_offsets[1], torch.zeros(3)))
        self.assertTrue(torch.allclose(obj_offsets[4], torch.zeros(3)))
        self.assertTrue(torch.allclose(hand_offsets[0], torch.zeros(3)))
        self.assertTrue(torch.allclose(hand_offsets[1], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(hand_offsets[2], obj_offsets[2]))
        self.assertTrue(torch.allclose(hand_offsets[3], obj_offsets[3]))
        self.assertTrue(torch.allclose(hand_offsets[4], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(hand_offsets[5], torch.zeros(3)))
        self.assertAlmostEqual(info["start_jump_norm"], 0.3, places=6)
        self.assertAlmostEqual(info["end_jump_norm"], 0.6, places=6)

    def test_interaction_alignment_hand_offsets_are_mesh_space(self):
        obj_trans = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ], dtype=torch.float32)
        hand_trans = torch.zeros_like(obj_trans)

        obj_offsets, hand_offsets, _ = compute_interaction_alignment_offsets(
            obj_trans,
            hand_trans,
            i0=1,
            i1=3,
            max_offset=10.0,
            gamma=1.0,
        )

        flat = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        mano_param_offsets = hand_offsets @ flat
        self.assertTrue(torch.allclose(obj_offsets[1], torch.tensor([-1.0, -2.0, -3.0])))
        self.assertTrue(torch.allclose(mano_param_offsets[1] @ flat, hand_offsets[1]))

    def test_grasp_template_anchor_loss_uses_dense_template_anchor(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]]],
                           dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        confidence = torch.tensor([[0.9]], dtype=torch.float32)
        anchor_face = torch.tensor([[0]], dtype=torch.long)
        anchor_bary = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

        loss, info = compute_grasp_template_anchor_loss(
            hand,
            obj,
            faces,
            confidence,
            anchor_face,
            anchor_bary,
            high_conf_thresh=0.8,
            gap_start=0.005,
            gap_clamp=0.05,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(info["n_template_active"], 1)
        self.assertAlmostEqual(info["mean_template_conf"], 0.9, places=5)
        self.assertTrue(torch.equal(info["template_active_count"], torch.tensor([1.0])))
        self.assertAlmostEqual(float(info["template_gap_mean"][0]), 0.03, places=5)
        self.assertAlmostEqual(float(info["template_conf_mean"][0]), 0.9, places=5)

    def test_grasp_template_anchor_loss_reports_frame_level_info(self):
        hand = torch.tensor([
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ], dtype=torch.float32)
        obj = torch.tensor([
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]],
            [[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]],
        ], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        confidence = torch.tensor([[0.9, 0.8], [0.7, 0.1]], dtype=torch.float32)
        anchor_face = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
        anchor_bary = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=torch.float32)

        _, info = compute_grasp_template_anchor_loss(
            hand,
            obj,
            faces,
            confidence,
            anchor_face,
            anchor_bary,
            high_conf_thresh=0.6,
            gap_start=0.005,
        )

        self.assertTrue(torch.equal(info["template_active_count"], torch.tensor([2.0, 1.0])))
        self.assertAlmostEqual(float(info["template_gap_mean"][0]), 0.02, places=5)
        self.assertAlmostEqual(float(info["template_gap_mean"][1]), 0.03, places=5)
        self.assertAlmostEqual(float(info["template_conf_mean"][0]), 0.85, places=5)
        self.assertAlmostEqual(float(info["template_conf_mean"][1]), 0.7, places=5)

    def test_grasp_template_anchor_loss_ignores_low_confidence(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]]],
                           dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        loss, info = compute_grasp_template_anchor_loss(
            hand,
            obj,
            faces,
            torch.tensor([[0.2]], dtype=torch.float32),
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32),
            high_conf_thresh=0.8,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(info["n_template_active"], 0)

    def test_neighbor_template_bridge_loss_only_uses_nearby_unreliable_frames(self):
        hand = torch.zeros(4, 1, 3, dtype=torch.float32)
        obj = torch.tensor([[[0.10, 0.0, 0.0], [0.0, 0.10, 0.0], [0.0, 0.0, 0.10]]],
                           dtype=torch.float32).repeat(4, 1, 1)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        template_conf = torch.ones(4, 1, dtype=torch.float32)
        template_face = torch.zeros(4, 1, dtype=torch.long)
        template_bary = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32).repeat(4, 1, 1)
        reliable = torch.tensor([False, False, True, False])

        loss, info = compute_neighbor_template_bridge_loss(
            hand,
            obj,
            faces,
            template_conf,
            template_face,
            template_bary,
            reliable,
            max_bridge_distance=1,
            high_conf_thresh=0.05,
            gap_start=0.0,
            gap_clamp=0.2,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(info["n_bridge_frames"], 2)
        self.assertEqual(info["n_bridge_active"], 2)

    def test_grasp_template_anchor_loss_scales_object_gradient_per_axis(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.02, 0.03, 0.04], [0.0, 0.03, 0.04], [0.02, 0.0, 0.04]]],
                           dtype=torch.float32, requires_grad=True)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        loss, _ = compute_grasp_template_anchor_loss(
            hand,
            obj,
            faces,
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32),
            high_conf_thresh=0.8,
            gap_start=0.0,
            gap_clamp=0.1,
            obj_grad_scale=(0.0, 0.0, 1.0),
        )
        loss.backward()

        self.assertAlmostEqual(float(obj.grad[0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(obj.grad[0, 0, 1]), 0.0, places=6)
        self.assertNotEqual(float(obj.grad[0, 0, 2]), 0.0)

    def test_gap_closing_uses_only_high_confidence_contacts(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        loss, info = compute_confident_gap_closing_loss(
            hand,
            obj,
            faces,
            self._cache(torch.tensor([[0.9, 0.2]])),
            high_conf_thresh=0.8,
            gap_start=0.005,
            gap_clamp=0.05,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(info["n_gap_active"], 1)

    def test_gap_closing_ignores_small_gaps(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        cache = ContactCache(
            bucket=torch.tensor([[2]], dtype=torch.int8),
            signed_dist=torch.zeros(1, 1),
            face_id_topk=torch.tensor([[[0]]], dtype=torch.long),
            bary_topk=torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32),
            weight_topk=torch.ones(1, 1, 1),
            active_idx=torch.tensor([[0]], dtype=torch.long),
            active_mask=torch.tensor([[True]]),
            argmax_face=torch.tensor([[0]], dtype=torch.long),
            contact_confidence=torch.tensor([[1.0]]),
        )

        loss, info = compute_confident_gap_closing_loss(
            hand, obj, faces, cache, high_conf_thresh=0.8, gap_start=0.005)

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(info["n_gap_active"], 0)

    def test_gap_closing_scales_object_gradient_per_axis(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.02, 0.03, 0.04], [0.0, 0.03, 0.04], [0.02, 0.0, 0.04]]],
                           dtype=torch.float32, requires_grad=True)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        cache = ContactCache(
            bucket=torch.tensor([[2]], dtype=torch.int8),
            signed_dist=torch.zeros(1, 1),
            face_id_topk=torch.tensor([[[0]]], dtype=torch.long),
            bary_topk=torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32),
            weight_topk=torch.ones(1, 1, 1),
            active_idx=torch.tensor([[0]], dtype=torch.long),
            active_mask=torch.tensor([[True]]),
            argmax_face=torch.tensor([[0]], dtype=torch.long),
            contact_confidence=torch.tensor([[1.0]]),
        )

        loss, _ = compute_confident_gap_closing_loss(
            hand,
            obj,
            faces,
            cache,
            high_conf_thresh=0.8,
            gap_start=0.0,
            gap_clamp=0.1,
            obj_grad_scale=(0.0, 0.0, 1.0),
        )
        loss.backward()

        self.assertAlmostEqual(float(obj.grad[0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(obj.grad[0, 0, 1]), 0.0, places=6)
        self.assertNotEqual(float(obj.grad[0, 0, 2]), 0.0)

    def test_patch_centroid_loss_uses_frame_level_contact_patch(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.02, 0.0, 0.0], [0.02, 0.02, 0.0], [0.02, 0.0, 0.02]]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        loss, info = compute_contact_patch_centroid_loss(
            hand,
            obj,
            faces,
            self._cache(torch.tensor([[0.9, 0.9]])),
            high_conf_thresh=0.8,
            min_contacts=2,
            gap_start=0.005,
            gap_clamp=0.05,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(info["n_patch_active"], 1)
        self.assertEqual(info["mean_patch_contacts"], 2.0)

    def test_patch_centroid_loss_requires_enough_contacts(self):
        hand = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]]], dtype=torch.float32)
        obj = torch.tensor([[[0.02, 0.0, 0.0], [0.02, 0.02, 0.0], [0.02, 0.0, 0.02]]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

        loss, info = compute_contact_patch_centroid_loss(
            hand,
            obj,
            faces,
            self._cache(torch.tensor([[0.9, 0.2]])),
            high_conf_thresh=0.8,
            min_contacts=2,
        )

        self.assertEqual(float(loss), 0.0)
        self.assertEqual(info["n_patch_active"], 0)


if __name__ == "__main__":
    unittest.main()
