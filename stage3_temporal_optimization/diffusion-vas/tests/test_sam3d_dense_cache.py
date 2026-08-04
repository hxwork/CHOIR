"""Tests for SAM3D dense pose cache helpers."""

import tempfile
import unittest
from pathlib import Path

import torch

from sam3d_sparse_keyframes import (
    build_sam3d_dense_cache_meta,
    load_sam3d_dense_pose_cache,
    sam3d_dense_cache_matches,
    save_sam3d_dense_pose_cache,
)


class Sam3dDenseCacheTest(unittest.TestCase):
    def test_cache_meta_matches_same_inputs_and_rejects_changed_indices(self):
        meta = build_sam3d_dense_cache_meta(
            sampled_indices=[0, 3, 7],
            reference_sampled_idx=0,
            onset_sampled_idx=1,
            end_sampled_exclusive=3,
            interaction_segment_lo=1,
            interaction_segment_hi=3,
            outlier_filter=True,
            outlier_max_angle_deg=60.0,
            outlier_max_iters=3,
            retry_count=3,
            global_frame_offset=0,
            seed=42,
            lambda_temp=0.5,
            config_path="/tmp/pipeline.yaml",
        )

        self.assertTrue(
            sam3d_dense_cache_matches(
                meta,
                sampled_indices=[0, 3, 7],
                reference_sampled_idx=0,
                onset_sampled_idx=1,
                end_sampled_exclusive=3,
                interaction_segment_lo=1,
                interaction_segment_hi=3,
                outlier_filter=True,
                outlier_max_angle_deg=60.0,
                outlier_max_iters=3,
                retry_count=3,
                global_frame_offset=0,
            )
        )
        self.assertFalse(
            sam3d_dense_cache_matches(
                meta,
                sampled_indices=[0, 4, 7],
                reference_sampled_idx=0,
                onset_sampled_idx=1,
                end_sampled_exclusive=3,
                interaction_segment_lo=1,
                interaction_segment_hi=3,
                outlier_filter=True,
                outlier_max_angle_deg=60.0,
                outlier_max_iters=3,
                retry_count=3,
                global_frame_offset=0,
            )
        )
        self.assertFalse(
            sam3d_dense_cache_matches(
                meta,
                sampled_indices=[0, 3, 7],
                reference_sampled_idx=0,
                onset_sampled_idx=1,
                end_sampled_exclusive=3,
                interaction_segment_lo=1,
                interaction_segment_hi=3,
                outlier_filter=True,
                outlier_max_angle_deg=60.0,
                outlier_max_iters=3,
                retry_count=0,
                global_frame_offset=0,
            )
        )

    def test_save_and_load_pose_snapshots_round_trip_tensors(self):
        snapshots = [
            {
                "sampled_idx": 1,
                "clip_frame_idx": 3,
                "pose": {
                    "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "translation": torch.tensor([0.1, 0.2, 0.3]),
                    "scale": torch.tensor([1.0, 1.0, 1.0]),
                },
            }
        ]
        meta = build_sam3d_dense_cache_meta(
            sampled_indices=[0, 3, 7],
            reference_sampled_idx=0,
            onset_sampled_idx=1,
            end_sampled_exclusive=3,
            interaction_segment_lo=1,
            interaction_segment_hi=3,
            outlier_filter=True,
            outlier_max_angle_deg=60.0,
            outlier_max_iters=3,
            retry_count=3,
            global_frame_offset=0,
            seed=42,
            lambda_temp=0.5,
            config_path=None,
        )
        summary = {"policy": "test", "entries": [{"sampled_idx": 1}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_sam3d_dense_pose_cache(Path(tmpdir), meta, summary, snapshots)
            loaded_summary, loaded_snapshots = load_sam3d_dense_pose_cache(
                Path(tmpdir),
                sampled_indices=[0, 3, 7],
                reference_sampled_idx=0,
                onset_sampled_idx=1,
                end_sampled_exclusive=3,
                interaction_segment_lo=1,
                interaction_segment_hi=3,
                outlier_filter=True,
                outlier_max_angle_deg=60.0,
                outlier_max_iters=3,
                retry_count=3,
                global_frame_offset=0,
            )

        self.assertEqual(loaded_summary["policy"], "test")
        self.assertEqual(loaded_snapshots[0]["sampled_idx"], 1)
        torch.testing.assert_close(
            loaded_snapshots[0]["pose"]["rotation"],
            snapshots[0]["pose"]["rotation"],
        )


if __name__ == "__main__":
    unittest.main()
