import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pack_data_for_baselines import _write_combined_label_masks, pack_one, resolve_video_ids


class PackDataForBaselinesTest(unittest.TestCase):
    def test_writes_combined_label_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "input"
            output_root = root / "output"
            video_dir = input_root / "VID"
            (video_dir / "obj_masks").mkdir(parents=True)
            (video_dir / "rh_masks").mkdir()
            (video_dir / "lh_masks").mkdir()
            (video_dir / "rgbs").mkdir()

            Image.new("RGB", (2, 2), "black").save(video_dir / "rgbs" / "000001.png")
            self._save_mask(video_dir / "obj_masks" / "000001.png", [(0, 1)])
            self._save_mask(video_dir / "rh_masks" / "000001.png", [(1, 0)])
            self._save_mask(video_dir / "lh_masks" / "000001.png", [(1, 1)])

            _write_combined_label_masks(
                video_dir / "rgbs",
                video_dir / "obj_masks",
                video_dir / "rh_masks",
                video_dir / "lh_masks",
                output_root / "VID" / "masks",
            )

            out = np.asarray(Image.open(output_root / "VID" / "masks" / "000001.png"))
            self.assertEqual(out[0, 0], 0)
            self.assertEqual(out[0, 1], 50)
            self.assertEqual(out[1, 0], 150)
            self.assertEqual(out[1, 1], 250)

    def test_resolve_video_ids_can_select_all_img_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir)
            (input_root / "IMG_0002").mkdir()
            (input_root / "VID").mkdir()
            (input_root / "IMG_0001").mkdir()

            video_ids = resolve_video_ids(input_root, [], all_img=True)

            self.assertEqual(video_ids, ["IMG_0001", "IMG_0002"])

    def test_pack_one_writes_rgb_and_mask_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "input"
            output_root = root / "output"
            video_id = "IMG_0001"
            video_dir = input_root / video_id
            (video_dir / "obj_masks").mkdir(parents=True)
            (video_dir / "rh_masks").mkdir()
            (video_dir / "rgbs").mkdir()

            for frame_idx in range(2):
                name = f"{frame_idx:06d}.png"
                Image.new("RGB", (4, 4), (frame_idx * 80, 20, 30)).save(
                    video_dir / "rgbs" / name
                )
                self._save_mask(video_dir / "obj_masks" / name, [(1, 1)])
                self._save_mask(video_dir / "rh_masks" / name, [(0, 0)])

            notes = pack_one(video_id, input_root, output_root, dry_run=False, fps=5)

            self.assertTrue((output_root / video_id / f"{video_id}.mp4").is_file())
            self.assertTrue((output_root / video_id / f"{video_id}_mask.mp4").is_file())
            self.assertFalse((output_root / video_id / "masks").exists())
            self.assertEqual(
                self._video_frame_count(output_root / video_id / f"{video_id}.mp4"),
                2,
            )
            self.assertEqual(
                self._video_frame_count(output_root / video_id / f"{video_id}_mask.mp4"),
                2,
            )
            self.assertIn(
                f"ok {video_id}: wrote {output_root / video_id}",
                notes,
            )

    def _save_mask(self, path: Path, coords: list[tuple[int, int]]) -> None:
        arr = np.zeros((2, 2, 4), dtype=np.uint8)
        for y, x in coords:
            arr[y, x] = (255, 255, 255, 255)
        Image.fromarray(arr, mode="RGBA").save(path)

    def _video_frame_count(self, path: Path) -> int:
        cap = cv2.VideoCapture(str(path))
        try:
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            return count
        finally:
            cap.release()

if __name__ == "__main__":
    unittest.main()
