import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


STAGE3_ROOT = Path(__file__).resolve().parents[1] / "stage3_temporal_optimization" / "diffusion-vas"
if str(STAGE3_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE3_ROOT))


class HO3DEvalExportTests(unittest.TestCase):
    def test_saves_magic_hoi_eval_data_schema_with_mano_joint_order(self):
        from choir_stage3.io.ho3d_eval_export import save_magic_hoi_eval_data

        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        openpose_to_mano = [mano_to_openpose.index(i) for i in range(21)]

        hand_joints_openpose = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
        hand_verts = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        obj_verts = (100 + np.arange(2 * 5 * 3, dtype=np.float32)).reshape(2, 5, 3)
        hand_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        obj_faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
        intrinsics = np.eye(3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "eval_data.npy"
            save_magic_hoi_eval_data(
                output_path=output_path,
                seq_name="hold_MC1_ho3d.0",
                intrinsics=intrinsics,
                hand_verts=hand_verts,
                hand_joints_openpose=hand_joints_openpose,
                hand_faces=hand_faces,
                obj_verts=obj_verts,
                obj_faces=obj_faces,
            )

            data = np.load(output_path, allow_pickle=True).item()

        expected_joints = hand_joints_openpose[:, openpose_to_mano, :]
        expected_hand_root = expected_joints[:, 0, :]
        expected_obj_root = obj_verts.mean(axis=1)

        self.assertEqual(data["full_seq_name"], "hold_MC1_ho3d.0")
        self.assertEqual(data["fnames"].tolist(), ["rgb/0000.png", "rgb/0001.png"])
        np.testing.assert_allclose(data["K"], intrinsics[None])
        np.testing.assert_allclose(data["jnts.right"].numpy(), expected_joints)
        np.testing.assert_allclose(data["root.right"].numpy(), expected_hand_root)
        np.testing.assert_allclose(data["j3d_ra.right"].numpy(), expected_joints - expected_hand_root[:, None, :])
        np.testing.assert_allclose(data["root.object"].numpy(), expected_obj_root)
        np.testing.assert_allclose(data["v3d_ra.object"].numpy(), obj_verts - expected_obj_root[:, None, :])
        np.testing.assert_allclose(data["v3d_right.object"].numpy(), obj_verts - expected_hand_root[:, None, :])
        self.assertTrue(torch.equal(data["faces"]["right"], torch.from_numpy(hand_faces)))
        self.assertTrue(torch.equal(data["faces"]["object"], torch.from_numpy(obj_faces)))


if __name__ == "__main__":
    unittest.main()
