"""Policy helpers for SAM3D-reset window selection.

This module is intentionally lightweight so the reset policy can be unit tested
without importing the full rendering/SAM3D stack.
"""


def choose_reset_bridge_start(
    c_frame: int,
    last_anchor_idx: int,
    trigger_reason: str,
    max_bridge_pre_frames: int = 8,
    iou_drop_margin: int = 2,
) -> int:
    """Choose the first frame for a local sequential SAM3D bridge.

    PnP remains the primary tracker; SAM3D should only cover a short suspicious
    window before the reset frame.  IoU-drop triggers usually localize the bad
    frame well, so use a small margin.  Rotation-disagreement probes are sparse,
    so use a bounded lookback window.
    """
    if c_frame <= last_anchor_idx:
        return c_frame

    reason = (trigger_reason or "").lower()
    if "iou dropped" in reason or "iou<" in reason or "consecutive frames" in reason or "angle_jump" in reason:
        pre_frames = max(0, int(iou_drop_margin))
    else:
        pre_frames = max(0, int(max_bridge_pre_frames))

    return max(last_anchor_idx + 1, c_frame - pre_frames)


def choose_reset_bridge_anchor(
    c_frame: int,
    last_anchor_idx: int,
    sam_cache: dict | None,
    max_anchor_angle_deg: float = 15.0,
    min_anchor_iou_delta: float = -0.03,
) -> int:
    """Pick the nearest trusted SAM3D probe before ``c_frame`` as bridge anchor.

    A local bridge must avoid a large first hop.  Sparse probes before reset can
    provide intermediate SAM3D trajectories, but only if they look consistent
    with PnP and do not visibly degrade silhouette IoU.  Flipped probes (e.g.
    ~180 deg disagreement) are rejected even if their silhouette IoU is similar.
    """
    best_anchor = int(last_anchor_idx)
    if not sam_cache:
        return best_anchor

    for fi in sorted(sam_cache):
        if fi <= last_anchor_idx or fi >= c_frame:
            continue
        entry = sam_cache[fi]
        angle = float(entry.get("angle_deg", float("inf")))
        iou_sam = float(entry.get("iou_sam", float("-inf")))
        iou_pnp = float(entry.get("iou_pnp", float("inf")))
        if angle <= max_anchor_angle_deg and (iou_sam - iou_pnp) >= min_anchor_iou_delta:
            best_anchor = int(fi)
    return best_anchor


def choose_reset_bridge_plan(
    c_frame: int,
    last_anchor_idx: int,
    trigger_reason: str,
    sam_cache: dict | None,
    validated_static_anchor_idx: int | None = None,
    max_bridge_pre_frames: int = 8,
    iou_drop_margin: int = 2,
    max_anchor_angle_deg: float = 15.0,
    min_anchor_iou_delta: float = -0.03,
) -> dict:
    """Return chain and replacement ranges for a reset bridge.

    Important distinction:
      * ``chain_anchor`` is a true SAM3D temporal anchor. It can be the previous
        reset/warm-up anchor, or a stage-aware static-skip anchor that was
        explicitly validated at runtime. One-shot sparse probes are not allowed
        to become temporal anchors.
      * ``replace_start`` is the first frame whose PnP pose should be replaced.
        Trusted probes can expand this local replacement window, but they do not
        become the temporal chain anchor.
    """
    trusted_probe = choose_reset_bridge_anchor(
        c_frame=c_frame,
        last_anchor_idx=last_anchor_idx,
        sam_cache=sam_cache,
        max_anchor_angle_deg=max_anchor_angle_deg,
        min_anchor_iou_delta=min_anchor_iou_delta,
    )
    replace_start = choose_reset_bridge_start(
        c_frame=c_frame,
        last_anchor_idx=last_anchor_idx,
        trigger_reason=trigger_reason,
        max_bridge_pre_frames=max_bridge_pre_frames,
        iou_drop_margin=iou_drop_margin,
    )
    if trusted_probe > last_anchor_idx:
        replace_start = min(replace_start, trusted_probe + 1)

    chain_anchor = int(last_anchor_idx)
    static_anchor = None
    if validated_static_anchor_idx is not None:
        static_anchor_i = int(validated_static_anchor_idx)
        if last_anchor_idx < static_anchor_i < c_frame:
            chain_anchor = static_anchor_i
            static_anchor = static_anchor_i
            replace_start = min(replace_start, static_anchor_i + 1)

    return {
        "chain_anchor": int(chain_anchor),
        "chain_start": int(chain_anchor + 1),
        "replace_start": int(replace_start),
        "trusted_probe_anchor": int(trusted_probe),
        "static_anchor": static_anchor,
    }


def should_replace_bridge_frame(
    iou_sam: float,
    iou_pnp: float,
    tolerance: float = 0.03,
    angle_to_trusted_deg: float | None = None,
    max_angle_to_trusted_deg: float = 90.0,
    use_iou_gate: bool = False,
) -> bool:
    """Return whether a SAM3D bridge pose is safe to write over PnP.

    The bridge may still be useful for reaching the reset frame, but individual
    pre-reset frames should not be overwritten when SAM3D has jumped to a
    different symmetric solution.  IoU gating is optional because SAM3D's
    translation can be biased, making IoU a poor proxy for rotation quality.
    """
    if angle_to_trusted_deg is not None and angle_to_trusted_deg > max_angle_to_trusted_deg:
        return False
    if use_iou_gate and float(iou_sam) < float(iou_pnp) - float(tolerance):
        return False
    return True
