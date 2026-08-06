import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def load_check_assets():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "check_assets.py"
    spec = importlib.util.spec_from_file_location("check_assets", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckAssetsTests(unittest.TestCase):
    def test_manifest_separates_internal_external_and_manual_assets(self):
        check_assets = load_check_assets()

        by_key = {asset.key: asset for asset in check_assets.ASSETS}

        self.assertEqual(by_key["tasterob_hoi_detector"].category, "internal")
        self.assertEqual(by_key["gfm_checkpoint"].category, "internal")
        self.assertEqual(by_key["mano_right_dynhamr_models"].category, "manual")
        self.assertEqual(by_key["bmc_convex_hulls"].category, "manual")
        self.assertEqual(by_key["diffusion_vas_amodal"].category, "external")
        self.assertTrue(by_key["sam3d_pipeline"].required)

    def test_collect_asset_status_reports_missing_required_assets(self):
        check_assets = load_check_assets()

        missing = check_assets.collect_asset_status(Path(self.id()))
        missing_by_key = {item.asset.key: item for item in missing}

        self.assertEqual(missing_by_key["sam2_hiera_large"].asset.category, "external")
        self.assertEqual(missing_by_key["tasterob_hoi_detector"].asset.category, "internal")
        self.assertIn(
            "stage1_preprocess/Yolov8/sam2/checkpoints/sam2.1_hiera_large.pt",
            missing_by_key["sam2_hiera_large"].target,
        )
        self.assertIn("Download with", missing_by_key["sam2_hiera_large"].hint)
        self.assertIn("choir_release_assets.zip", missing_by_key["tasterob_hoi_detector"].hint)

    def test_exit_code_ignores_optional_missing_assets(self):
        check_assets = load_check_assets()

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = check_assets.main(["--repo-root", self.id()])
        output = stdout.getvalue()

        self.assertEqual(exit_code, 1)
        self.assertIn("Missing required CHOIR assets", output)
        self.assertIn("sam3d_pipeline", output)


if __name__ == "__main__":
    unittest.main()
