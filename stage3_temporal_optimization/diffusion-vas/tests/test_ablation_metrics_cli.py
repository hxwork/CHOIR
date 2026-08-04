import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "compute_ablation_metrics.py"


class AblationMetricsCliTest(unittest.TestCase):
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
            "--ablation_root",
            "--mesh_root",
            "--data_root",
            "--output_dir",
            "--ablation",
            "--video_id",
            "--device",
            "--iou_batch_size",
            "--chamfer_samples",
            "--no_render_iou",
            "--no_flip_xy_for_iou",
        }
        self.assertTrue(expected_flags.issubset(flag_names))
        self.assertNotIn("--fps", flag_names)

    def test_acceleration_units_are_per_frame_squared(self):
        constants = {
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("cm/frame^2", constants)

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


if __name__ == "__main__":
    unittest.main()
