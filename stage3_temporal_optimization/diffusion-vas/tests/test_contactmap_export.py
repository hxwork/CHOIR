import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from contact_constraint import ContactCache
from hoi_io import (
    contact_cache_to_dense_hand_confidence,
    contact_cache_to_object_vertex_confidence,
    compute_distance_contact_confidence,
    interpolate_vertex_confidence,
    save_contactmap_sequence,
)


class ContactmapExportTest(unittest.TestCase):
    def _cache(self):
        return ContactCache(
            bucket=torch.tensor([[2, 2]], dtype=torch.int8),
            signed_dist=torch.zeros(1, 2),
            face_id_topk=torch.tensor([[[0], [0]]], dtype=torch.long),
            bary_topk=torch.tensor([[[[0.2, 0.3, 0.5]], [[0.9, 0.1, 0.0]]]], dtype=torch.float32),
            weight_topk=torch.ones(1, 2, 1),
            active_idx=torch.tensor([[1, 3]], dtype=torch.long),
            active_mask=torch.tensor([[True, False]]),
            argmax_face=torch.tensor([[0, -1]], dtype=torch.long),
            contact_confidence=torch.tensor([[0.8, 0.4]], dtype=torch.float32),
            observed_contact_confidence=torch.tensor([[0.25, 0.1]], dtype=torch.float32),
            raw_observed_contact_confidence=torch.tensor([[0.0, 0.2, 0.6, 0.0, 0.9]], dtype=torch.float32),
            raw_geometry_contact_confidence=torch.tensor([[0.1, 0.2, 0.6, 0.3, 0.9]], dtype=torch.float32),
        )

    def test_contact_cache_to_dense_hand_confidence_uses_raw_temporal_confidence(self):
        dense = contact_cache_to_dense_hand_confidence(self._cache(), num_hand_verts=5)

        self.assertEqual(dense.shape, (1, 5))
        self.assertAlmostEqual(float(dense[0, 1]), 0.8, places=6)
        self.assertEqual(float(dense[0, 3]), 0.0)

    def test_contact_cache_to_dense_hand_confidence_can_use_observed_confidence(self):
        dense = contact_cache_to_dense_hand_confidence(
            self._cache(),
            num_hand_verts=5,
            confidence_attr="observed_contact_confidence",
        )

        self.assertAlmostEqual(float(dense[0, 1]), 0.25, places=6)
        self.assertEqual(float(dense[0, 3]), 0.0)

    def test_contact_cache_to_dense_hand_confidence_returns_dense_raw_field(self):
        dense = contact_cache_to_dense_hand_confidence(
            self._cache(),
            num_hand_verts=5,
            confidence_attr="raw_observed_contact_confidence",
        )

        np.testing.assert_allclose(dense[0], np.array([0.0, 0.2, 0.6, 0.0, 0.9], dtype=np.float32))

    def test_contact_cache_to_object_vertex_confidence_scatters_barycentric_confidence(self):
        obj_faces = np.array([[0, 1, 2]], dtype=np.int64)

        obj_conf = contact_cache_to_object_vertex_confidence(self._cache(), obj_faces, num_obj_verts=3)

        self.assertEqual(obj_conf.shape, (1, 3))
        np.testing.assert_allclose(obj_conf[0], np.array([0.16, 0.24, 0.4], dtype=np.float32), atol=1e-6)

    def test_contact_cache_to_object_vertex_confidence_can_use_observed_confidence(self):
        obj_faces = np.array([[0, 1, 2]], dtype=np.int64)

        obj_conf = contact_cache_to_object_vertex_confidence(
            self._cache(),
            obj_faces,
            num_obj_verts=3,
            confidence_attr="observed_contact_confidence",
        )

        np.testing.assert_allclose(obj_conf[0], np.array([0.05, 0.075, 0.125], dtype=np.float32), atol=1e-6)

    def test_contact_cache_to_object_vertex_confidence_projects_dense_raw_to_nearest_object_vertex(self):
        hand_verts = np.array([[
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.9],
        ]], dtype=np.float32)
        obj_verts = np.array([[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]], dtype=np.float32)
        obj_faces = np.array([[0, 1, 2]], dtype=np.int64)

        obj_conf = contact_cache_to_object_vertex_confidence(
            self._cache(),
            obj_faces,
            num_obj_verts=3,
            confidence_attr="raw_observed_contact_confidence",
            hand_verts=hand_verts,
            obj_verts=obj_verts,
        )

        np.testing.assert_allclose(obj_conf[0], np.array([0.2, 0.6, 0.9], dtype=np.float32), atol=1e-6)

    def test_save_contactmap_sequence_uses_paper_rendering_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_verts = np.zeros((1, 3, 3), dtype=np.float32)
            hand_verts = np.zeros((1, 5, 3), dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            hand_conf = np.zeros((1, 5), dtype=np.float32)
            obj_conf = np.zeros((1, 3), dtype=np.float32)

            out_dir = save_contactmap_sequence(
                video_id="IMG_TEST",
                obj_verts_seq=obj_verts,
                obj_faces=faces,
                hand_verts_seq=hand_verts,
                hand_faces=faces,
                hand_confidence_seq=hand_conf,
                obj_confidence_seq=obj_conf,
                rendering_root=tmp,
            )

            root = Path(tmp) / "IMG_TEST_contactmap"
            self.assertEqual(Path(out_dir), root)
            self.assertTrue((root / "hand" / "0000_hand.ply").exists())
            self.assertTrue((root / "object" / "0000_mesh.ply").exists())

    def test_save_contactmap_sequence_accepts_sequence_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_verts = np.zeros((1, 3, 3), dtype=np.float32)
            hand_verts = np.zeros((1, 5, 3), dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            hand_conf = np.zeros((1, 5), dtype=np.float32)
            obj_conf = np.zeros((1, 3), dtype=np.float32)

            out_dir = save_contactmap_sequence(
                video_id="IMG_TEST",
                obj_verts_seq=obj_verts,
                obj_faces=faces,
                hand_verts_seq=hand_verts,
                hand_faces=faces,
                hand_confidence_seq=hand_conf,
                obj_confidence_seq=obj_conf,
                rendering_root=tmp,
                sequence_suffix="contactmap_perframe",
            )

            self.assertEqual(Path(out_dir), Path(tmp) / "IMG_TEST_contactmap_perframe")

    def test_interpolate_vertex_confidence_expands_sparse_frames_to_full_sequence(self):
        sparse_conf = np.array([[0.0, 0.0], [1.0, 0.5], [0.0, 0.0]], dtype=np.float32)

        full_conf = interpolate_vertex_confidence([0, 2, 4], sparse_conf, num_frames=5)

        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.25],
                [1.0, 0.5],
                [0.5, 0.25],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(full_conf, expected, atol=1e-6)

    def test_compute_distance_contact_confidence_uses_two_cm_scale(self):
        hand_verts = np.array([[
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ]], dtype=np.float32)
        obj_verts = np.array([[
            [0.0, 0.0, 0.0],
            [0.04, 0.0, 0.0],
        ]], dtype=np.float32)

        hand_conf, obj_conf = compute_distance_contact_confidence(
            hand_verts,
            obj_verts,
            distance_scale=0.02,
        )

        expected_mid = np.exp(-1.0)
        np.testing.assert_allclose(hand_conf[0], np.array([1.0, expected_mid, 1.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obj_conf[0], np.array([1.0, 1.0], dtype=np.float32), atol=1e-6)

    def test_compute_distance_contact_confidence_splats_nearest_object_with_max(self):
        hand_verts = np.array([[
            [0.01, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [1.02, 0.0, 0.0],
        ]], dtype=np.float32)
        obj_verts = np.array([[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]], dtype=np.float32)

        _, obj_conf = compute_distance_contact_confidence(
            hand_verts,
            obj_verts,
            distance_scale=0.02,
        )

        expected_obj0 = np.exp(-((0.01 / 0.02) ** 2))
        expected_obj1 = np.exp(-1.0)
        np.testing.assert_allclose(obj_conf[0], np.array([expected_obj0, expected_obj1], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
