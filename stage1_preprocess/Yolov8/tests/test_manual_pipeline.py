from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import run_with_manual_object_mask as pipeline
import run_with_hoi_detector as hoi_pipeline


class ManualPipelineTest(unittest.TestCase):

    def test_load_input_videos_filters_before_creating_output_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            (data_dir / "selected.mp4").touch()
            (data_dir / "ignored.mp4").touch()
            (data_dir / "nested").mkdir()
            (data_dir / "nested" / "nested.mp4").touch()

            videos = pipeline.load_input_videos(
                data_dir,
                output_dir,
                video_ids={"selected"},
            )

            self.assertEqual([video.stem for video in videos], ["selected"])
            self.assertTrue((output_dir / "selected").is_dir())
            self.assertFalse((output_dir / "ignored").exists())
            self.assertFalse((output_dir / "nested").exists())

    def test_save_frames_as_video_preserves_1080_height(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            pipeline.save_frames_as_video([frame], output_path, fps=30)

            capture = cv2.VideoCapture(str(output_path))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()

            self.assertEqual((width, height), (1920, 1080))

    def test_visualization_is_enabled_by_default_and_can_be_disabled(self):
        self.assertTrue(hasattr(pipeline, "build_arg_parser"))

        parser = pipeline.build_arg_parser()

        self.assertTrue(parser.parse_args([]).visualize)
        self.assertFalse(parser.parse_args(["--no_visualize"]).visualize)

    def test_hoi_visualization_preserves_1080_height(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            hoi_pipeline.visualize_sam_tracking(
                [frame],
                {},
                {},
                {},
                {},
                output_path,
                pause_frame_idx=-1,
            )

            capture = cv2.VideoCapture(str(output_path))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()

            self.assertEqual((width, height), (1920, 1080))

    def test_hoi_visualization_is_enabled_by_default_and_can_be_disabled(self):
        self.assertTrue(hasattr(hoi_pipeline, "build_arg_parser"))

        parser = hoi_pipeline.build_arg_parser()

        self.assertTrue(parser.parse_args([]).visualize)
        self.assertFalse(parser.parse_args(["--no_visualize"]).visualize)


if __name__ == "__main__":
    unittest.main()
