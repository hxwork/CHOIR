import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_paper_hoi_video import (
    compute_crop_camera,
    find_hoi_sequence_dir,
    infer_frame_offset,
    list_video_dirs,
    parse_rgb_triplet,
    parse_args,
)


class RenderPaperHoiVideoTest(unittest.TestCase):
    def test_list_video_dirs_keeps_requested_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_B").mkdir()
            (root / "IMG_A").mkdir()
            (root / "notes.txt").write_text("ignore me")

            selected = list_video_dirs(root, ["IMG_A", "IMG_B"])

            self.assertEqual([p.name for p in selected], ["IMG_A", "IMG_B"])

    def test_list_video_dirs_rejects_missing_requested_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "MISSING"):
                list_video_dirs(Path(tmp), ["MISSING"])

    def test_find_hoi_sequence_dir_prefers_contact_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq_dir = Path(tmp)
            contact = seq_dir / "optimized_hoi_contact_seq"
            plain = seq_dir / "optimized_hoi_seq"
            contact.mkdir()
            plain.mkdir()

            self.assertEqual(find_hoi_sequence_dir(seq_dir), contact)

    def test_compute_crop_camera_scales_intrinsics_for_requested_resolution(self):
        intrinsics = np.array(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        bbox = [460, 290, 1460, 790]

        camera = compute_crop_camera(intrinsics, bbox, height=1024, width=2048)

        self.assertEqual(camera.image_size, (1024, 2048))
        self.assertAlmostEqual(camera.focal_length[0], 2048.0)
        self.assertAlmostEqual(camera.focal_length[1], 2048.0)
        self.assertAlmostEqual(camera.principal_point[0], 1024.0)
        self.assertAlmostEqual(camera.principal_point[1], 512.0)

    def test_parse_rgb_triplet_validates_range(self):
        self.assertEqual(parse_rgb_triplet(["0.1", "0.2", "0.3"]), (0.1, 0.2, 0.3))
        with self.assertRaisesRegex(ValueError, "range"):
            parse_rgb_triplet(["1.2", "0.2", "0.3"])

    def test_default_resolution_is_1080p_final_quality(self):
        config = parse_args(["--video_id", "IMG_A"])

        self.assertEqual(config.height, 1080)
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.quality, "final")
        self.assertEqual(config.image_scale, 2)

    def test_preview_quality_uses_fast_render_scale(self):
        config = parse_args(["--video_id", "IMG_A", "--quality", "preview"])

        self.assertEqual(config.quality, "preview")
        self.assertEqual(config.image_scale, 1)

    def test_infer_frame_offset_reads_first_mano_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            mano_dir = Path(tmp) / "mano_params"
            mano_dir.mkdir()
            (mano_dir / "21.json").write_text("{}")
            (mano_dir / "22.json").write_text("{}")

            self.assertEqual(infer_frame_offset(Path(tmp)), 21)


if __name__ == "__main__":
    unittest.main()
