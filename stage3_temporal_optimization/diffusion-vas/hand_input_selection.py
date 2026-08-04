import json
import os
from dataclasses import dataclass
from typing import Optional


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


def _side_file(seq_path: str, hand_side: str, suffix: str) -> Optional[str]:
    prefix = HAND_SIDE_TO_PREFIX[hand_side]
    return _existing_file(os.path.join(seq_path, f"{prefix}_{suffix}.json"))


def _side_masks_dir(seq_path: str, hand_side: str) -> Optional[str]:
    prefix = HAND_SIDE_TO_PREFIX[hand_side]
    return _existing_dir(os.path.join(seq_path, f"{prefix}_masks"))


def _single_hand_file(seq_path: str, suffix: str) -> Optional[str]:
    candidates = [
        path
        for prefix in ("lh", "rh")
        for path in [os.path.join(seq_path, f"{prefix}_{suffix}.json")]
        if os.path.exists(path)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates)[0]


def _single_hand_masks_dir(seq_path: str) -> Optional[str]:
    candidates = [
        path
        for prefix in ("lh", "rh")
        for path in [os.path.join(seq_path, f"{prefix}_masks")]
        if os.path.isdir(path)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates)[0]


def resolve_hand_inputs(seq_path, hand_side: Optional[str] = None) -> HandInputs:
    """Resolve per-sequence hand inputs while preserving legacy single-hand data."""
    seq_path = os.fspath(seq_path)
    _validate_hand_side(hand_side)

    mano_root_dir = os.path.join(seq_path, "mano_params")
    export_meta_path = os.path.join(mano_root_dir, "export_meta.json")

    if os.path.exists(export_meta_path):
        with open(export_meta_path, "r") as f:
            export_meta = json.load(f)

        if export_meta.get("mode") == "dual":
            if hand_side is None:
                seq_name = os.path.basename(seq_path)
                raise ValueError(
                    f"Sequence {seq_name} has dual MANO params; "
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
                bbox_file=_side_file(seq_path, hand_side, "bbox"),
                keypoints_file=_side_file(seq_path, hand_side, "keypoints"),
                hand_masks_dir=_side_masks_dir(seq_path, hand_side),
                is_dual=True,
            )

    bbox_file = _side_file(seq_path, hand_side, "bbox") if hand_side else _single_hand_file(seq_path, "bbox")
    keypoints_file = (
        _side_file(seq_path, hand_side, "keypoints")
        if hand_side
        else _single_hand_file(seq_path, "keypoints")
    )
    hand_masks_dir = _side_masks_dir(seq_path, hand_side) if hand_side else _single_hand_masks_dir(seq_path)
    return HandInputs(
        selected_hand_side=hand_side,
        mano_params_dir=mano_root_dir,
        bbox_file=bbox_file,
        keypoints_file=keypoints_file,
        hand_masks_dir=hand_masks_dir,
        is_dual=False,
    )
