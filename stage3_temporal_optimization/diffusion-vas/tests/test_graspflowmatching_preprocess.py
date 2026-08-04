import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gfm_preprocess import export_graspflowmatching_sequence


class GraspFlowMatchingPreprocessTest(unittest.TestCase):
    def test_exports_sparse_sequence_with_real_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "IMG_TEST"
            verts = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            obj_rot = np.stack([np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)])
            obj_trans = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
            obj_scale = np.array([1.5, 1.5, 1.5], dtype=np.float32)
            mano_root_orient = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32)
            mano_pose = np.zeros((2, 15, 3), dtype=np.float32)
            mano_trans = np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float32)
            is_right = np.array([[1.0], [0.0]], dtype=np.float32)

            output_dir = export_graspflowmatching_sequence(
                seq_path=seq_path,
                sampled_indices=[0, 7],
                canonical_verts=verts,
                canonical_faces=faces,
                obj_rot_mats=obj_rot,
                obj_trans=obj_trans,
                obj_scale=obj_scale,
                mano_root_orient=mano_root_orient,
                mano_pose=mano_pose,
                mano_trans=mano_trans,
                is_right=is_right,
            )

            self.assertEqual(output_dir, seq_path / "optimized_hoi_seq")
            self.assertTrue((output_dir / "obj_canonical.obj").is_file())
            self.assertTrue((output_dir / "obj_00000.json").is_file())
            self.assertTrue((output_dir / "mano_00007.json").is_file())
            self.assertFalse((output_dir / "mano_00001.json").exists())

            with (output_dir / "obj_00007.json").open() as f:
                obj_data = json.load(f)
            self.assertEqual(obj_data["translation"], [0.4, 0.5, 0.6])
            self.assertEqual(obj_data["scale"], [1.5, 1.5, 1.5])

            with (output_dir / "mano_00007.json").open() as f:
                mano_data = json.load(f)
            self.assertEqual(mano_data["root_orient"], [0.04, 0.05, 0.06])
            self.assertEqual(mano_data["trans"], [1.0, 1.1, 1.2])
            self.assertEqual(mano_data["is_right"], 0.0)

    def test_requires_matching_frame_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "IMG_TEST"
            with self.assertRaisesRegex(ValueError, "sampled_indices length"):
                export_graspflowmatching_sequence(
                    seq_path=seq_path,
                    sampled_indices=[0, 1],
                    canonical_verts=np.zeros((3, 3), dtype=np.float32),
                    canonical_faces=np.zeros((1, 3), dtype=np.int64),
                    obj_rot_mats=np.eye(3, dtype=np.float32)[None],
                    obj_trans=np.zeros((1, 3), dtype=np.float32),
                    obj_scale=np.ones(3, dtype=np.float32),
                    mano_root_orient=np.zeros((1, 3), dtype=np.float32),
                    mano_pose=np.zeros((1, 15, 3), dtype=np.float32),
                    mano_trans=np.zeros((1, 3), dtype=np.float32),
                    is_right=np.ones((1, 1), dtype=np.float32),
                )

    def test_can_export_clean_dedicated_sequence_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "IMG_TEST"
            stale_dir = seq_path / "grasp_correction" / "gfm_input_hoi_seq"
            stale_dir.mkdir(parents=True)
            (stale_dir / "mano_99999.json").write_text("{}", encoding="utf-8")

            output_dir = export_graspflowmatching_sequence(
                seq_path=seq_path,
                sampled_indices=[3],
                canonical_verts=np.zeros((3, 3), dtype=np.float32),
                canonical_faces=np.zeros((1, 3), dtype=np.int64),
                obj_rot_mats=np.eye(3, dtype=np.float32)[None],
                obj_trans=np.zeros((1, 3), dtype=np.float32),
                obj_scale=np.ones(3, dtype=np.float32),
                mano_root_orient=np.zeros((1, 3), dtype=np.float32),
                mano_pose=np.zeros((1, 15, 3), dtype=np.float32),
                mano_trans=np.zeros((1, 3), dtype=np.float32),
                is_right=np.ones((1, 1), dtype=np.float32),
                output_dir_name="grasp_correction/gfm_input_hoi_seq",
                clean_output=True,
            )

            self.assertEqual(output_dir, stale_dir)
            self.assertTrue((output_dir / "mano_00003.json").is_file())
            self.assertFalse((output_dir / "mano_99999.json").exists())


if __name__ == "__main__":
    unittest.main()
