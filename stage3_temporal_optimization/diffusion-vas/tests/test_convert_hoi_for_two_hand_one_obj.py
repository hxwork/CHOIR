import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convert_hoi_for_two_hand_one_obj import (
    combine_hand_meshes,
    estimate_rigid_transform,
    estimate_similarity_transform,
    load_mesh,
    merge_sequences,
    transform_vertices,
    write_ply,
)


def rotation_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class ConvertHoiForTwoHandOneObjTest(unittest.TestCase):
    def test_estimate_rigid_transform_recovers_known_transform(self):
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_rotation = rotation_z(math.radians(30.0))
        expected_translation = np.array([0.4, -0.2, 1.5], dtype=np.float64)
        target = transform_vertices(source, expected_rotation, expected_translation)

        rotation, translation, rms_error = estimate_rigid_transform(source, target)

        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-7)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-7)
        self.assertLess(rms_error, 1e-10)

    def test_estimate_similarity_transform_recovers_scale_rotation_translation(self):
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected_scale = 2.5
        expected_rotation = rotation_z(math.radians(20.0))
        expected_translation = np.array([-0.3, 0.4, 1.2], dtype=np.float64)
        target = expected_scale * transform_vertices(source, expected_rotation, np.zeros(3)) + expected_translation

        scale, rotation, translation, rms_error = estimate_similarity_transform(source, target)

        self.assertAlmostEqual(scale, expected_scale, places=7)
        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-7)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-7)
        self.assertLess(rms_error, 1e-10)

    def test_combine_hand_meshes_offsets_right_faces(self):
        left_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        left_faces = np.array([[0, 1, 2]], dtype=np.int32)
        right_vertices = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32)
        right_faces = np.array([[0, 2, 1]], dtype=np.int32)

        vertices, faces = combine_hand_meshes(left_vertices, left_faces, right_vertices, right_faces)

        np.testing.assert_array_equal(vertices, np.vstack([left_vertices, right_vertices]))
        np.testing.assert_array_equal(faces, np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int32))

    def test_merge_sequences_transforms_right_hand_into_left_object_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_seq = root / "left"
            right_seq = root / "right"
            output_seq = root / "merged"
            for seq in (left_seq, right_seq):
                (seq / "hand").mkdir(parents=True)
                (seq / "object").mkdir(parents=True)

            object_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            hand_faces = np.array([[0, 1, 2]], dtype=np.int32)
            left_object = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            right_to_left_rotation = rotation_z(math.radians(90.0))
            right_to_left_translation = np.array([2.0, 3.0, 4.0], dtype=np.float64)
            right_object = (left_object - right_to_left_translation) @ right_to_left_rotation
            left_hand = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32)
            right_hand = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32)
            expected_right_hand = transform_vertices(right_hand, right_to_left_rotation, right_to_left_translation)

            write_ply(left_seq / "object" / "0000_mesh.ply", left_object, object_faces)
            write_ply(left_seq / "hand" / "0000_hand.ply", left_hand, hand_faces)
            write_ply(right_seq / "object" / "0000_mesh.ply", right_object, object_faces)
            write_ply(right_seq / "hand" / "0000_hand.ply", right_hand, hand_faces)

            summary = merge_sequences(left_seq, right_seq, output_seq)

            self.assertEqual(summary.frame_count, 1)
            merged_right, _ = load_mesh(output_seq / "right_hand" / "0000_hand.ply")
            np.testing.assert_allclose(merged_right, expected_right_hand, atol=1e-6)
            combined_vertices, combined_faces = load_mesh(output_seq / "hand" / "0000_hand.ply")
            self.assertEqual(combined_vertices.shape[0], 6)
            np.testing.assert_array_equal(combined_faces, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32))

            metadata = json.loads((output_seq / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_count"], 1)
            self.assertLess(metadata["alignment_rms"]["max"], 1e-6)

    def test_merge_sequences_uses_middle_object_scale_for_scaled_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_seq = root / "left"
            right_seq = root / "right"
            output_seq = root / "merged"
            for seq in (left_seq, right_seq):
                (seq / "hand").mkdir(parents=True)
                (seq / "object").mkdir(parents=True)

            object_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
            hand_faces = np.array([[0, 1, 2]], dtype=np.int32)
            right_object = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            left_object = right_object * 4.0
            left_center = left_object.mean(axis=0)
            right_hand = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.1, 0.0]], dtype=np.float32)
            left_hand = right_hand * 4.0
            expected_object = left_center + 0.5 * (left_object - left_center)
            expected_left_hand = left_center + 0.5 * (left_hand - left_center)
            expected_right_hand = left_center + 0.5 * (4.0 * right_hand - left_center)

            write_ply(left_seq / "object" / "0000_mesh.ply", left_object, object_faces)
            write_ply(left_seq / "hand" / "0000_hand.ply", left_hand, hand_faces)
            write_ply(right_seq / "object" / "0000_mesh.ply", right_object, object_faces)
            write_ply(right_seq / "hand" / "0000_hand.ply", right_hand, hand_faces)

            summary = merge_sequences(left_seq, right_seq, output_seq)

            self.assertEqual(summary.frame_count, 1)
            merged_object, _ = load_mesh(output_seq / "object" / "0000_mesh.ply")
            merged_left, _ = load_mesh(output_seq / "left_hand" / "0000_hand.ply")
            merged_right, _ = load_mesh(output_seq / "right_hand" / "0000_hand.ply")
            np.testing.assert_allclose(merged_object, expected_object, atol=1e-6)
            np.testing.assert_allclose(merged_left, expected_left_hand, atol=1e-6)
            np.testing.assert_allclose(merged_right, expected_right_hand, atol=1e-6)

            metadata = json.loads((output_seq / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["scale_mode"], "middle")
            self.assertAlmostEqual(metadata["similarity_scale"]["mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
