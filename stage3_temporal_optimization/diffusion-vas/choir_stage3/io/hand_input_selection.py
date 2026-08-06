"""Resolve per-sequence hand inputs using the canonical output layout."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from choir_stage3.io.video_layout import VideoLayout


HAND_SIDE_TO_PREFIX = {
    "left": "lh",
    "right": "rh",
}


@dataclass(frozen=True)
class HandInputs:
    selected_hand_side: Optional[str]
    mano_params_dir: str
    bbox_file: Optional[str]
    keypoints_file: Optional[str]
    hand_masks_dir: Optional[str]
    is_dual: bool


def _validate_hand_side(hand_side: Optional[str]) -> None:
    if hand_side is not None and hand_side not in HAND_SIDE_TO_PREFIX:
        raise ValueError("hand_side must be either 'left' or 'right'")


def _existing_file(path: str) -> Optional[str]:
    return path if os.path.exists(path) else None


def _existing_dir(path: str) -> Optional[str]:
    return path if os.path.isdir(path) else None


def _side_file(layout: VideoLayout, hand_side: str, kind: str) -> Optional[str]:
    if kind == "bbox":
        return _existing_file(str(layout.bbox_json(hand_side)))
    if kind == "keypoints":
        return _existing_file(str(layout.keypoints_json(hand_side)))
    raise ValueError(kind)


def _side_masks_dir(layout: VideoLayout, hand_side: str) -> Optional[str]:
    return _existing_dir(str(layout.hand_masks_dir(hand_side)))


def _single_hand_file(layout: VideoLayout, kind: str) -> Optional[str]:
    candidates = [
        path
        for side in ("left", "right")
        for path in [_side_file(layout, side, kind)]
        if path is not None
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates)[0]


def _single_hand_masks_dir(layout: VideoLayout) -> Optional[str]:
    candidates = [
        path
        for side in ("left", "right")
        for path in [_side_masks_dir(layout, side)]
        if path is not None
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates)[0]


def resolve_hand_inputs(seq_path, hand_side: Optional[str] = None) -> HandInputs:
    """Resolve per-sequence hand inputs for the selected side."""
    layout = VideoLayout.from_root(seq_path)
    _validate_hand_side(hand_side)

    mano_root_dir = str(layout.mano_params_dir)
    export_meta_path = os.path.join(mano_root_dir, "export_meta.json")

    if os.path.exists(export_meta_path):
        with open(export_meta_path, "r") as f:
            export_meta = json.load(f)

        if export_meta.get("mode") == "dual":
            if hand_side is None:
                raise ValueError(
                    f"Sequence {layout.video_id} has dual MANO params; "
                    "pass --hand_side left or --hand_side right."
                )

            local_mano_dir = os.path.join(mano_root_dir, hand_side)
            meta_mano_dir = (
                export_meta.get("hands", {})
                .get(hand_side, {})
                .get("mano_params_dir")
            )
            mano_params_dir = local_mano_dir if os.path.isdir(local_mano_dir) else meta_mano_dir
            if not mano_params_dir or not os.path.isdir(mano_params_dir):
                raise FileNotFoundError(
                    f"Missing MANO params directory for {hand_side} hand: {local_mano_dir}"
                )

            return HandInputs(
                selected_hand_side=hand_side,
                mano_params_dir=mano_params_dir,
                bbox_file=_side_file(layout, hand_side, "bbox"),
                keypoints_file=_side_file(layout, hand_side, "keypoints"),
                hand_masks_dir=_side_masks_dir(layout, hand_side),
                is_dual=True,
            )

    bbox_file = _side_file(layout, hand_side, "bbox") if hand_side else _single_hand_file(layout, "bbox")
    keypoints_file = (
        _side_file(layout, hand_side, "keypoints")
        if hand_side
        else _single_hand_file(layout, "keypoints")
    )
    hand_masks_dir = _side_masks_dir(layout, hand_side) if hand_side else _single_hand_masks_dir(layout)
    return HandInputs(
        selected_hand_side=hand_side,
        mano_params_dir=mano_root_dir,
        bbox_file=bbox_file,
        keypoints_file=keypoints_file,
        hand_masks_dir=hand_masks_dir,
        is_dual=False,
    )
