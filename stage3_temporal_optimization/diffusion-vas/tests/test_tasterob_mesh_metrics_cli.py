import ast
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "compute_tasterob_mesh_metrics.py"
WRAPPER_PATH = REPO_ROOT / "run_tasterob_mesh_metrics.sh"


def load_module():
    spec = importlib.util.spec_from_file_location("compute_tasterob_mesh_metrics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TasteRobMeshMetricsCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    def test_cli_exposes_expected_arguments(self):
        flag_names = {
            arg.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }

        expected_flags = {
            "--render_root",
            "--mesh_root",
            "--data_root",
            "--output_dir",
            "--video_id",
            "--device",
            "--iou_batch_size",
            "--chamfer_samples",
            "--no_render_iou",
            "--no_flip_xy_for_iou",
            "--unit_to_cm",
        }
        self.assertTrue(expected_flags.issubset(flag_names))
        self.assertNotIn("--ablation", flag_names)

    def test_metric_keys_are_declared(self):
        metric_keys = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "METRIC_KEYS" for target in node.targets):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                metric_keys = {
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                }

        self.assertEqual(
            metric_keys,
            {"mIoU", "Pen. Ratio", "H-O Dist.", "Acc_h", "Acc_o"},
        )

    def test_discovers_only_render_ids_with_base_mesh_dirs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_root = tmp_path / "rendered_output" / "ours_TasteRob"
            mesh_root = tmp_path / "ours_SelfCaptured"
            for name in ["594", "IMG_5140", "IMG_5140_contactmap", "hold_ABF12_ho3d", "notes.txt"]:
                path = render_root / name
                if "." in name:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)
            for name in ["594", "IMG_5140", "IMG_5140_contactmap"]:
                (mesh_root / name / "hand").mkdir(parents=True, exist_ok=True)
                (mesh_root / name / "object").mkdir(parents=True, exist_ok=True)
            (mesh_root / "IMG_5140" / "hand" / "0000_hand.ply").write_text("ply\n", encoding="utf-8")
            (mesh_root / "IMG_5140" / "object" / "0000_mesh.ply").write_text("ply\n", encoding="utf-8")
            (mesh_root / "594" / "hand" / "0000_hand.ply").write_text("ply\n", encoding="utf-8")
            (mesh_root / "594" / "object" / "0000_mesh.ply").write_text("ply\n", encoding="utf-8")

            runs, warnings = module.discover_runs(render_root, mesh_root, None)

        self.assertEqual([run.video_id for run in runs], ["594", "IMG_5140"])
        self.assertEqual(warnings, [])

    def test_interaction_range_prefers_render_then_data_then_all_frames(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_seq_dir = tmp_path / "render"
            data_seq_dir = tmp_path / "data"
            render_seq_dir.mkdir()
            data_seq_dir.mkdir()
            (data_seq_dir / "stage_frame_ranges.json").write_text(
                '{"3_interaction": {"original_start": 2, "original_end": 5}}',
                encoding="utf-8",
            )
            self.assertEqual(module.load_interaction_frames(render_seq_dir, data_seq_dir, 10)[0], [2, 3, 4])

            (render_seq_dir / "stage_frame_ranges.json").write_text(
                '{"3_interaction": {"original_start": 1, "original_end": 3}}',
                encoding="utf-8",
            )
            self.assertEqual(module.load_interaction_frames(render_seq_dir, data_seq_dir, 10)[0], [1, 2])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            render_seq_dir = tmp_path / "render"
            data_seq_dir = tmp_path / "data"
            render_seq_dir.mkdir()
            data_seq_dir.mkdir()
            self.assertEqual(module.load_interaction_frames(render_seq_dir, data_seq_dir, 3)[0], [0, 1, 2])

    def test_wrapper_points_to_tasterob_metrics_script(self):
        wrapper = WRAPPER_PATH.read_text(encoding="utf-8")

        self.assertIn('SCRIPT="compute_tasterob_mesh_metrics.py"', wrapper)
        self.assertIn("RENDER_ROOT", wrapper)
        self.assertIn("MESH_ROOT", wrapper)
        self.assertIn("DATA_ROOT", wrapper)
        self.assertIn("--video_id", wrapper)


if __name__ == "__main__":
    unittest.main()
