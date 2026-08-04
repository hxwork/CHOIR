import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from save_modal_amodal_overlay_jpgs import (
    AMODAL_COLOR,
    compute_mask_bbox,
    ENV_INPUT_DATA,
    MODAL_COLOR,
    crop_and_resize_frame,
    default_data_root,
    load_raw_frames,
    make_bbox_skeleton_panel,
    overlay_mask_with_color,
    parse_args,
    project_crop_mask_to_original_frame,
    save_overlay_comparisons,
    select_hand_annotations_for_side,
)


class SaveModalAmodalOverlayJpgsTest(unittest.TestCase):
    def test_overlay_uses_distinct_modal_and_amodal_colors(self):
        rgb = np.ones((8, 8, 3), dtype=np.float32) * 0.2
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        modal = overlay_mask_with_color(rgb, mask, MODAL_COLOR, boundary_thickness=0)
        amodal = overlay_mask_with_color(rgb, mask, AMODAL_COLOR, boundary_thickness=0)

        self.assertFalse(np.allclose(modal, amodal))
        self.assertTrue(np.all(modal >= 0.0))
        self.assertTrue(np.all(modal <= 1.0))

    def test_crop_and_resize_frame_matches_expected_output_shape(self):
        frame = np.ones((20, 30, 3), dtype=np.uint8) * 255
        cropped = crop_and_resize_frame(frame, [-5, 0, 25, 20], (10, 20), is_mask=False)

        self.assertEqual(cropped.shape, (10, 20, 3))
        self.assertEqual(cropped.dtype, np.uint8)

    def test_save_overlay_comparisons_writes_side_by_side_jpgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "mask_overlay_compare_jpgs"
            rgbs = np.ones((2, 8, 8, 3), dtype=np.float32) * 0.25
            modal_masks = np.zeros((2, 8, 8), dtype=np.uint8)
            amodal_masks = np.zeros((2, 8, 8), dtype=np.uint8)
            modal_masks[:, 2:5, 2:5] = 1
            amodal_masks[:, 1:7, 1:7] = 1

            save_overlay_comparisons(rgbs, modal_masks, amodal_masks, out_dir, overwrite=True)

            saved_files = sorted(out_dir.glob("*.jpg"))
            self.assertEqual([p.name for p in saved_files], ["000000.jpg", "000001.jpg"])
            saved = imageio.imread(saved_files[0])
            self.assertEqual(saved.shape[:2], (8, 34))
            self.assertEqual(saved.shape[2], 3)
            self.assertGreater(np.abs(saved[:, :8].astype(int) - saved[:, -8:].astype(int)).sum(), 0)

    def test_save_overlay_comparisons_accepts_third_bbox_skeleton_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "mask_overlay_compare_jpgs"
            rgbs = np.ones((1, 8, 8, 3), dtype=np.float32) * 0.25
            modal_masks = np.zeros((1, 8, 8), dtype=np.uint8)
            amodal_masks = np.zeros((1, 8, 8), dtype=np.uint8)
            bbox_panel = np.ones((1, 8, 8, 3), dtype=np.float32) * 0.5
            modal_masks[:, 2:5, 2:5] = 1
            amodal_masks[:, 1:7, 1:7] = 1

            save_overlay_comparisons(
                rgbs,
                modal_masks,
                amodal_masks,
                out_dir,
                bbox_skeleton_panels=bbox_panel,
                overwrite=True,
            )

            saved = imageio.imread(out_dir / "000000.jpg")
            self.assertEqual(saved.shape, (8, 60, 3))

    def test_load_raw_frames_reads_masks_without_external_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp) / "obj_masks"
            mask_dir.mkdir()
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[2:4, 3:6] = 255
            cv2.imwrite(str(mask_dir / "0.png"), mask)

            loaded = load_raw_frames(mask_dir, frame_type="mask")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].dtype, np.uint8)
            self.assertEqual(int(loaded[0].sum()), 6)

    def test_project_crop_mask_to_original_frame_inverts_global_bbox_crop(self):
        raw_mask = np.zeros((24, 32), dtype=np.uint8)
        raw_mask[6:18, 8:24] = 1
        bbox = [0, 0, 32, 12]
        crop_mask = crop_and_resize_frame(raw_mask, bbox, (256, 512), is_mask=True)

        projected = project_crop_mask_to_original_frame(crop_mask, bbox, raw_mask.shape)

        self.assertEqual(projected.shape, raw_mask.shape)
        np.testing.assert_array_equal(projected[:12], raw_mask[:12])
        self.assertEqual(int(projected[12:].sum()), 0)

    def test_compute_mask_bbox_returns_tight_box(self):
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[3:8, 5:11] = 1

        self.assertEqual(compute_mask_bbox(mask), (5, 3, 11, 8))
        self.assertIsNone(compute_mask_bbox(np.zeros_like(mask)))

    def test_select_hand_annotations_for_side(self):
        both = {
            "lh": {"bbox": {"0": [1, 1, 5, 5]}, "keypoints": {}},
            "rh": {"bbox": {"0": [10, 10, 20, 20]}, "keypoints": {}},
        }
        self.assertEqual(
            select_hand_annotations_for_side(both, "rh"),
            {"rh": both["rh"]},
        )
        self.assertEqual(
            select_hand_annotations_for_side(both, "lh"),
            {"lh": both["lh"]},
        )
        only_lh = {"lh": both["lh"]}
        self.assertEqual(select_hand_annotations_for_side(only_lh, "rh"), only_lh)

    def test_make_bbox_skeleton_panel_draws_annotations(self):
        rgb = np.ones((32, 48, 3), dtype=np.float32) * 0.25
        obj_mask = np.zeros((32, 48), dtype=np.uint8)
        obj_mask[8:20, 12:30] = 1
        keypoints = [[20 + i, 10 + (i % 5)] for i in range(21)]
        hand_annotations = {
            "rh": {
                "bbox": {"0": [18, 8, 42, 28]},
                "keypoints": {"0": keypoints},
            }
        }

        panel = make_bbox_skeleton_panel(rgb, obj_mask, 0, hand_annotations)
        panel_rh = make_bbox_skeleton_panel(rgb, obj_mask, 0, hand_annotations, hand_side="rh")

        self.assertEqual(panel.shape, rgb.shape)
        self.assertGreater(np.abs(panel - rgb).sum(), 0)
        np.testing.assert_allclose(panel, panel_rh)

    def test_parse_args_hand_side_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            args = parse_args(["--video_id", "X", "--data-root", str(tmp_path)])
            self.assertEqual(args.hand_side, "rh")

    def test_parse_args_hand_side_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            args = parse_args(["--video_id", "X", "--data-root", str(tmp_path), "--hand_side", "lh"])
            self.assertEqual(args.hand_side, "lh")

    def test_parse_args_video_id_and_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            args = parse_args(["--video_id", "MYSEQ", "--data-root", str(tmp_path)])
            self.assertEqual(args.video_id, ["MYSEQ"])
            self.assertEqual(args.data_root, tmp_path)

    def test_parse_args_accepts_multiple_video_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            args = parse_args(["--video_id", "A", "B", "--data-root", str(tmp_path)])
            self.assertEqual(args.video_id, ["A", "B"])
            self.assertEqual(args.data_root, tmp_path)

    def test_default_data_root_respects_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            old = os.environ.get(ENV_INPUT_DATA)
            try:
                os.environ[ENV_INPUT_DATA] = str(tmp_path)
                self.assertEqual(default_data_root(), tmp_path)
            finally:
                if old is None:
                    os.environ.pop(ENV_INPUT_DATA, None)
                else:
                    os.environ[ENV_INPUT_DATA] = old

    def test_cli_writes_jpgs_under_video_id_subfolder(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "save_modal_amodal_overlay_jpgs.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_id = "CLI_SMOKE"
            seq = root / video_id
            (seq / "rgbs").mkdir(parents=True)
            (seq / "obj_masks").mkdir(parents=True)
            (seq / "amodal_masks").mkdir(parents=True)

            for i in range(2):
                rgb = np.zeros((24, 32, 3), dtype=np.uint8)
                rgb[..., 0] = 50 + i * 10
                cv2.imwrite(str(seq / "rgbs" / f"{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[6:18, 8:24] = 255
                cv2.imwrite(str(seq / "obj_masks" / f"{i}.png"), mask)
                amodal = np.zeros((256, 512), dtype=np.uint8)
                amodal[80:180, 100:400] = 255
                cv2.imwrite(str(seq / "amodal_masks" / f"{i}.png"), amodal)

            with open(seq / "global_bbox.json", "w") as f:
                json.dump([[0, 0, 32, 12]] * 2, f)

            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--video_id",
                    video_id,
                    "--data-root",
                    str(root),
                    "--max_frames",
                    "1",
                    "--overwrite",
                ],
                cwd=str(repo_root),
                check=True,
            )

            out_jpg = seq / "mask_overlay_compare_jpgs" / "000000.jpg"
            self.assertTrue(out_jpg.is_file())
            img = imageio.imread(out_jpg)
            self.assertEqual(img.shape, (24, 132, 3))

    def test_cli_processes_multiple_video_ids(self):
        def _populate_sequence(seq: Path):
            (seq / "rgbs").mkdir(parents=True)
            (seq / "obj_masks").mkdir(parents=True)
            (seq / "amodal_masks").mkdir(parents=True)
            for i in range(2):
                rgb = np.zeros((24, 32, 3), dtype=np.uint8)
                rgb[..., 1] = 40 + i * 10
                cv2.imwrite(str(seq / "rgbs" / f"{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[6:18, 8:24] = 255
                cv2.imwrite(str(seq / "obj_masks" / f"{i}.png"), mask)
                amodal = np.zeros((256, 512), dtype=np.uint8)
                amodal[80:180, 120:380] = 255
                cv2.imwrite(str(seq / "amodal_masks" / f"{i}.png"), amodal)
            with open(seq / "global_bbox.json", "w") as f:
                json.dump([[0, 0, 32, 12]] * 2, f)

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "save_modal_amodal_overlay_jpgs.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1, v2 = "MULTI_A", "MULTI_B"
            _populate_sequence(root / v1)
            _populate_sequence(root / v2)

            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--video_id",
                    v1,
                    v2,
                    "--data-root",
                    str(root),
                    "--max_frames",
                    "1",
                    "--overwrite",
                ],
                cwd=str(repo_root),
                check=True,
            )

            for vid in (v1, v2):
                out_jpg = root / vid / "mask_overlay_compare_jpgs" / "000000.jpg"
                self.assertTrue(out_jpg.is_file(), msg=str(out_jpg))


if __name__ == "__main__":
    unittest.main()
