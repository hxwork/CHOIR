import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sam3d_ref_masks import load_sam3d_ref_obj_mask, resolve_sam3d_ref_obj_mask_path


class Sam3dRefMaskTest(unittest.TestCase):
    def test_resolves_ref_mask_from_obj_masks_not_sequence_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp)
            (seq / "obj_masks").mkdir()
            cv2.imwrite(str(seq / "0.png"), np.full((4, 4), 255, dtype=np.uint8))
            cv2.imwrite(str(seq / "obj_masks" / "3.png"), np.full((4, 4), 255, dtype=np.uint8))

            path = resolve_sam3d_ref_obj_mask_path(str(seq), 3)

            self.assertEqual(path, str(seq / "obj_masks" / "3.png"))

    def test_loads_binary_ref_mask_from_obj_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp)
            (seq / "obj_masks").mkdir()
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 255
            cv2.imwrite(str(seq / "obj_masks" / "2.png"), mask)

            loaded, path = load_sam3d_ref_obj_mask(str(seq), 2)

            self.assertEqual(path, str(seq / "obj_masks" / "2.png"))
            self.assertEqual(loaded.dtype, np.uint8)
            self.assertEqual(int(loaded.sum()), 4)


if __name__ == "__main__":
    unittest.main()
