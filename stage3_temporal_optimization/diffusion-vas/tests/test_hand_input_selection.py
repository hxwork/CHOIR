import json
import tempfile
import unittest
from pathlib import Path

from debug_bbox import load_hand_data
from hand_input_selection import resolve_hand_inputs


class HandInputSelectionTest(unittest.TestCase):
    def test_dual_requires_explicit_hand_side(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "E02"
            mano_dir = seq_path / "mano_params"
            mano_dir.mkdir(parents=True)
            (mano_dir / "export_meta.json").write_text(
                json.dumps({"mode": "dual", "hands": {"left": {}, "right": {}}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "pass --hand_side left or --hand_side right"):
                resolve_hand_inputs(seq_path)

    def test_dual_uses_selected_hand_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "E02"
            mano_dir = seq_path / "mano_params"
            (mano_dir / "left").mkdir(parents=True)
            (mano_dir / "right").mkdir(parents=True)
            (mano_dir / "export_meta.json").write_text(
                json.dumps({"mode": "dual", "hands": {"left": {}, "right": {}}}),
                encoding="utf-8",
            )
            (seq_path / "rh_keypoints.json").write_text("{}", encoding="utf-8")
            (seq_path / "rh_bbox.json").write_text("{}", encoding="utf-8")
            (seq_path / "rh_masks").mkdir()

            inputs = resolve_hand_inputs(seq_path, hand_side="right")

            self.assertTrue(inputs.is_dual)
            self.assertEqual(inputs.selected_hand_side, "right")
            self.assertEqual(inputs.mano_params_dir, str(mano_dir / "right"))
            self.assertEqual(inputs.keypoints_file, str(seq_path / "rh_keypoints.json"))
            self.assertEqual(inputs.bbox_file, str(seq_path / "rh_bbox.json"))
            self.assertEqual(inputs.hand_masks_dir, str(seq_path / "rh_masks"))

    def test_legacy_single_hand_uses_root_mano_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "OLD"
            mano_dir = seq_path / "mano_params"
            mano_dir.mkdir(parents=True)
            (mano_dir / "00000.json").write_text("{}", encoding="utf-8")
            (seq_path / "lh_keypoints.json").write_text("{}", encoding="utf-8")
            (seq_path / "lh_bbox.json").write_text("{}", encoding="utf-8")

            inputs = resolve_hand_inputs(seq_path)

            self.assertFalse(inputs.is_dual)
            self.assertIsNone(inputs.selected_hand_side)
            self.assertEqual(inputs.mano_params_dir, str(mano_dir))
            self.assertEqual(inputs.keypoints_file, str(seq_path / "lh_keypoints.json"))
            self.assertEqual(inputs.bbox_file, str(seq_path / "lh_bbox.json"))

    def test_load_hand_data_filters_to_selected_side(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir)
            (seq_path / "lh_bbox.json").write_text(json.dumps({"0": [1, 2, 3, 4]}), encoding="utf-8")
            (seq_path / "rh_bbox.json").write_text(json.dumps({"0": [5, 6, 7, 8]}), encoding="utf-8")
            (seq_path / "lh_keypoints.json").write_text(json.dumps({"0": [[1, 2]]}), encoding="utf-8")
            (seq_path / "rh_keypoints.json").write_text(json.dumps({"0": [[5, 6]]}), encoding="utf-8")

            frames = load_hand_data(seq_path, num_frames=1, hand_side="right")

            self.assertEqual(set(frames[0].keys()), {"rh"})
            self.assertEqual(frames[0]["rh"]["bbox"], [5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
