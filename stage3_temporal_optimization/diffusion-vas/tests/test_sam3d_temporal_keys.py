"""Tests for SAM3D temporal latent key policy."""

import ast
import pathlib
import unittest


def _constant_tuple(name: str) -> tuple[str, ...]:
    module_path = pathlib.Path(__file__).resolve().parents[1] / "sam3d_reset.py"
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in sam3d_reset.py")


class Sam3dTemporalKeysTest(unittest.TestCase):
    def test_geometry_freeze_keeps_shape_and_scale_only(self):
        freeze_keys = _constant_tuple("_SAM3D_FREEZE_KEYS")

        self.assertEqual(freeze_keys, ("shape", "scale"))
        self.assertNotIn("translation_scale", freeze_keys)

    def test_pose_guidance_keys_are_unchanged(self):
        guide_keys = _constant_tuple("_SAM3D_GUIDE_KEYS")

        self.assertEqual(guide_keys, ("6drotation_normalized", "translation"))


if __name__ == "__main__":
    unittest.main()
