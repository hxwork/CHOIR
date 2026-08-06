import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import labeling
import run_with_manual_object_mask as manual_pipeline
import run_with_hoi_detector as hoi_pipeline


class DataLayoutTest(unittest.TestCase):

    def load_layout(self):
        self.assertIsNotNone(importlib.util.find_spec("data_layout"))
        return importlib.import_module("data_layout")

    def test_discovers_only_flat_mp4_inputs(self):
        layout = self.load_layout()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "flat.mp4").touch()
            (data_dir / "nested").mkdir()
            (data_dir / "nested" / "nested.mp4").touch()

            videos = layout.discover_input_videos(data_dir)

            self.assertEqual(videos, [data_dir / "flat.mp4"])

    def test_filters_flat_videos_by_video_id(self):
        layout = self.load_layout()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            for video_id in ("a", "b"):
                (data_dir / f"{video_id}.mp4").touch()

            videos = layout.discover_input_videos(data_dir, {"b"})

            self.assertEqual(videos, [data_dir / "b.mp4"])

    def test_annotation_directory_is_under_output_video_directory(self):
        layout = self.load_layout()
        video_path = Path("/repo/data/demo.mp4")

        self.assertEqual(
            layout.annotation_dir_for_video(Path("/repo/output"), video_path),
            Path("/repo/output/demo/inputs/annotations"),
        )

    def test_visualization_directory_is_under_output_video_directory(self):
        layout = self.load_layout()
        video_path = Path("/repo/data/demo.mp4")

        self.assertEqual(
            layout.visualization_dir_for_video(Path("/repo/output"), video_path),
            Path("/repo/output/demo/stage1/visualizations"),
        )

    def test_hoi_loader_uses_flat_inputs_and_selected_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "downstream"
            for video_id in ("selected", "ignored"):
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / f"{video_id}.mp4").touch()

            output_paths = hoi_pipeline.load_my_data_videos(
                data_dir,
                output_dir,
                video_ids={"selected"},
            )

            source = data_dir / "selected.mp4"
            expected = output_dir / "selected" / "selected.mp4"
            self.assertEqual(output_paths, [source])
            self.assertTrue(expected.exists())
            self.assertFalse((output_dir / "ignored").exists())

    def test_all_entry_points_default_to_repository_data_directory(self):
        layout = self.load_layout()
        self.assertTrue(hasattr(labeling, "build_arg_parser"))

        labeling_args = labeling.build_arg_parser().parse_args([])
        manual_args = manual_pipeline.build_arg_parser().parse_args([])
        hoi_args = hoi_pipeline.build_arg_parser().parse_args([])

        self.assertEqual(Path(labeling_args.data), layout.DEFAULT_DATA_DIR)
        self.assertEqual(Path(manual_args.data_dir), layout.DEFAULT_DATA_DIR)
        self.assertEqual(Path(hoi_args.data_dir), layout.DEFAULT_DATA_DIR)
        self.assertEqual(Path(labeling_args.output), layout.DEFAULT_OUTPUT_DATA_DIR)
        self.assertEqual(Path(manual_args.output_dir), layout.DEFAULT_OUTPUT_DATA_DIR)
        self.assertEqual(Path(hoi_args.output_dir), layout.DEFAULT_OUTPUT_DATA_DIR)
        self.assertEqual(hoi_args.data_source, "local")


if __name__ == "__main__":
    unittest.main()
