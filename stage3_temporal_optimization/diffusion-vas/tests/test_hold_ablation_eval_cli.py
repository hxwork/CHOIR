import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "eval_ours.py"
ABLATION_SCRIPT = REPO_ROOT / "demo_fitting_5stages_sam3d_reset_ablation.py"
RUNNER_SCRIPT = REPO_ROOT / "run_hold_ablation_overnight.sh"


class HoldAblationEvalCliTest(unittest.TestCase):
    NEW_REBUTTAL_ABLATIONS = {
        "stage3_raw_metric",
        "post_gfm_metric",
    }

    def test_eval_ours_accepts_explicit_prediction_path(self):
        tree = ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))
        flag_names = {
            arg.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }

        self.assertIn("--pred_data_path", flag_names)

    def test_ablation_script_saves_eval_data(self):
        tree = ast.parse(ABLATION_SCRIPT.read_text(encoding="utf-8"))
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("save_ablation_eval_data", function_names)
        self.assertIn("eval_data.npy", constants)

    def test_rebuttal_metric_cutpoint_ablations_are_registered(self):
        tree = ast.parse(ABLATION_SCRIPT.read_text(encoding="utf-8"))
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        runner_text = RUNNER_SCRIPT.read_text(encoding="utf-8")

        for ablation in self.NEW_REBUTTAL_ABLATIONS:
            self.assertIn(ablation, constants)
            self.assertIn(ablation, runner_text)

    def test_hold_runner_declares_expected_sequences_and_ablations(self):
        text = RUNNER_SCRIPT.read_text(encoding="utf-8")
        expected_sequences = {
            "hold_ABF12_ho3d.180",
            "hold_ABF14_ho3d.180",
            "hold_GPMF12_ho3d.90",
            "hold_GPMF14_ho3d.90",
            "hold_MC1_ho3d.0",
            "hold_MC4_ho3d.0",
            "hold_MDF12_ho3d.60",
            "hold_MDF14_ho3d.300",
            "hold_ShSu10_ho3d.30",
            "hold_ShSu12_ho3d.30",
            "hold_SM2_ho3d.90",
            "hold_SM4_ho3d.0",
            "hold_SMu1_ho3d.0",
            "hold_SMu40_ho3d.0",
        }
        expected_ablations = {
            "full",
            "no_fp",
            "no_vp",
            "no_stage4_offset",
            "no_stage5_pen",
            "no_dyn_contact",
            "no_stage5_contact",
            "no_stage5_smooth",
        }

        for seq_name in expected_sequences:
            self.assertIn(seq_name, text)
        for ablation in expected_ablations:
            self.assertIn(ablation, text)
        self.assertIn("nvidia-smi", text)
        self.assertIn("failed_runs", text)
        self.assertIn("hold_ablation_metrics.json", text)
        self.assertIn("hold_ablation_metrics.csv", text)


if __name__ == "__main__":
    unittest.main()
