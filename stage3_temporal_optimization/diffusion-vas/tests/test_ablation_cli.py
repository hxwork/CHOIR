import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "demo_fitting_5stages_sam3d_reset_ablation.py"


class AblationCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    def test_cli_exposes_ablation_choices(self):
        expected_choices = {
            "full",
            "no_fp",
            "no_vp",
            "no_stage4_offset",
            "no_stage5_pen",
            "no_dyn_contact",
            "no_stage5_contact",
            "no_stage5_smooth",
        }

        choices = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            has_ablation_flag = any(
                isinstance(arg, ast.Constant) and arg.value == "--ablation"
                for arg in node.args
            )
            if not has_ablation_flag:
                continue
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                    choices = {
                        elt.value
                        for elt in keyword.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }

        self.assertEqual(choices, expected_choices)

    def test_fit_and_visualize_pose_accepts_ablation_argument(self):
        fn = next(
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "fit_and_visualize_pose"
        )

        self.assertIn("ablation", [arg.arg for arg in fn.args.args])


if __name__ == "__main__":
    unittest.main()
