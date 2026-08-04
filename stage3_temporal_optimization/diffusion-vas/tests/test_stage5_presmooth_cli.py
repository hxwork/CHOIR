import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "demo_fitting_5stages_sam3d_reset.py"


class Stage5PresmoothCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    def test_fit_and_visualize_pose_defaults_to_presmoothing_object(self):
        fn = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fit_and_visualize_pose"
        )
        defaults_by_arg = {
            arg.arg: default
            for arg, default in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)
        }

        self.assertIn("stage5_presmooth_object", defaults_by_arg)
        self.assertIsInstance(defaults_by_arg["stage5_presmooth_object"], ast.Constant)
        self.assertIs(defaults_by_arg["stage5_presmooth_object"].value, True)

    def test_fit_and_visualize_pose_defaults_to_no_sam3d_ref_keyframe_init(self):
        fn = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fit_and_visualize_pose"
        )
        defaults_by_arg = {
            arg.arg: default
            for arg, default in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)
        }

        self.assertIn("sam3d_ref_keyframe_init", defaults_by_arg)
        self.assertIsInstance(defaults_by_arg["sam3d_ref_keyframe_init"], ast.Constant)
        self.assertIs(defaults_by_arg["sam3d_ref_keyframe_init"].value, False)

    def test_fit_and_visualize_pose_defaults_to_applying_stage4_camera_offset(self):
        fn = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fit_and_visualize_pose"
        )
        defaults_by_arg = {
            arg.arg: default
            for arg, default in zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)
        }

        self.assertIn("no_stage4_camera_offset", defaults_by_arg)
        self.assertIsInstance(defaults_by_arg["no_stage4_camera_offset"], ast.Constant)
        self.assertIs(defaults_by_arg["no_stage4_camera_offset"].value, False)

    def test_cli_exposes_no_stage5_presmooth_object_flag(self):
        flag_names = [
            arg.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]

        self.assertIn("--no_stage5_presmooth_object", flag_names)

    def test_cli_exposes_no_stage4_camera_offset_flag(self):
        flag_names = [
            arg.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]

        self.assertIn("--no_stage4_camera_offset", flag_names)

    def test_worker_forwards_no_stage4_camera_offset_flag(self):
        fit_call = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "fit_and_visualize_pose"
        )
        keyword_names = {kw.arg for kw in fit_call.keywords}

        self.assertIn("no_stage4_camera_offset", keyword_names)

    def test_cli_exposes_sam3d_ref_keyframe_init_flag(self):
        flag_names = [
            arg.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]

        self.assertIn("--sam3d_ref_keyframe_init", flag_names)


if __name__ == "__main__":
    unittest.main()
