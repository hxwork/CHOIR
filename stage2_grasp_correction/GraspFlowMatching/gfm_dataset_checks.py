"""Small validation helpers for GraspFlowMatching sampling."""

from data_layout import DEFAULT_SOURCE_DIR


def assert_nonempty_dataset(dataset_len, selected_video_ids=None, source_dir=None):
    """Fail fast when a filtered test dataset has no samples."""
    if dataset_len > 0:
        return

    video_msg = ""
    if selected_video_ids:
        video_msg = f" for video_id(s): {', '.join(map(str, selected_video_ids))}"
    root = source_dir or str(DEFAULT_SOURCE_DIR)
    raise RuntimeError(
        "No GraspFlowMatching samples found"
        f"{video_msg}. Expected optimized_hoi_init_seq or optimized_hoi_seq with matching "
        f"mano_*.json and obj_*.json files under {root}/<video_id>."
    )
