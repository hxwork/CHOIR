import unittest

import torch

from stage5_object_lite import (
    DEFAULT_OBJECT_LITE_MAX_TRANS_DELTA,
    project_object_translation_delta_,
)


class Stage5ObjectLiteTest(unittest.TestCase):
    def test_default_translation_delta_allows_small_stage4_residuals(self):
        self.assertEqual(DEFAULT_OBJECT_LITE_MAX_TRANS_DELTA, 0.04)

    def test_projects_object_translation_to_default_delta_ball(self):
        trans = torch.tensor([[0.08, 0.0, 0.0]], dtype=torch.float32)
        anchor = torch.zeros_like(trans)

        project_object_translation_delta_(trans, anchor)

        self.assertAlmostEqual(float(torch.linalg.norm(trans - anchor).item()), 0.04, places=6)


if __name__ == "__main__":
    unittest.main()
