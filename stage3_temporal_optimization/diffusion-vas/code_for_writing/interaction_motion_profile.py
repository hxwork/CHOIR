"""Heuristic interaction motion typing for the 2–3 (interaction) segment.

Uses cheap 2D signals (CoTracker tracks + optional amodal mask shape) before
multi-frame PnP is available, and can optionally incorporate PnP rotation
step statistics after Stage 3 builds ``stage3_object_pose_init`` (PnP tail
plus optional SAM3D dense merge in ``demo_fitting_5stages_sam3d_reset``).

This is intentionally lightweight and tunable — for calibration on your data,
adjust ``DEFAULT_THRESHOLDS`` after inspecting the exported JSON fields.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _SciRotation
except ImportError:  # pragma: no cover
    _SciRotation = None

DEFAULT_THRESHOLDS = {
    # CoTracker: mean absolute 2D Kabsch rotation between consecutive frames.
    "cot_mean_deg_rotation_likely": 4.0,
    "cot_p90_deg_rotation_likely": 12.0,
    "cot_mean_deg_translation_likely": 2.0,
    "cot_p90_deg_translation_likely": 6.0,
    # Integrated / endpoint 2D cues (out-of-plane 3D rotation often shows as
    # many small per-step Kabsch angles that still sum to a large net change).
    "cot_cumulative_abs_deg_rotation_likely": 22.0,
    "cot_endpoint_abs_deg_rotation_likely": 18.0,
    # Mask principal-axis step (degrees), consecutive frames in interaction.
    "mask_mean_deg_rotation_hint": 2.5,
    "mask_max_deg_rotation_hint": 9.0,
    # PnP (optional): geodesic angle between consecutive col-major R in interaction.
    "pnp_mean_deg_rotation_likely": 6.0,
    "pnp_mean_deg_translation_likely": 3.5,
    # MANO global root in camera (HaMeR root_orient + T_w2c, same as TemporalHandObjectPose).
    "mano_mean_step_deg_rotation_likely": 7.0,
    "mano_p90_step_deg_rotation_likely": 18.0,
    "mano_cumulative_step_deg_rotation_hint": 40.0,
    "mano_endpoint_geodesic_deg_rotation_likely": 35.0,
    # CoTracker transform quality: rotation must explain the tracked point
    # motion substantially better than translation-only before long-window
    # cumulative/endpoint angles are allowed to trigger rotation_likely.
    "cot_rotation_quality_gain_likely": 0.35,
    "cot_rotation_quality_min_points": 8,
}


def _angle_between_rotations_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotation matrices (same convention as PnP col-major)."""
    R_rel = R_a.T @ R_b
    c = (float(np.trace(R_rel)) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _kabsch_2d_abs_angle_deg(p0: np.ndarray, p1: np.ndarray) -> float:
    """Absolute rotation angle from a 2D Kabsch alignment of matched points (K,2)."""
    if p0.shape[0] < 2 or p0.shape != p1.shape:
        return float("nan")
    c0 = p0.mean(axis=0)
    c1 = p1.mean(axis=0)
    x = p0 - c0
    y = p1 - c1
    s = x.T @ y
    u, _, vt = np.linalg.svd(s)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u_adj = u.copy()
        u_adj[:, -1] *= -1.0
        r = u_adj @ vt
    ang = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
    return abs(ang)


def _principal_axis_deg_from_mask(mask: np.ndarray) -> float:
    """Orientation (deg) of major axis of mask pixels; +x = 0, CCW positive."""
    m = np.asarray(mask).astype(bool)
    if m.sum() < 16:
        return float("nan")
    yy, xx = np.where(m)
    coords = np.stack([xx.astype(np.float64), yy.astype(np.float64)], axis=1)
    cov = np.cov(coords.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    return float(math.degrees(math.atan2(float(major[1]), float(major[0]))))


def _wrap_angle_deg(delta: float) -> float:
    """Map angle difference to [-90, 90] (principal axis has pi periodicity)."""
    x = (delta + 180.0) % 180.0
    if x > 90.0:
        x = 180.0 - x
    return x


def _cotracker_kabsch_series(
    tracks_xy: np.ndarray,
    tracks_vis: np.ndarray | None,
    interaction_lo: int,
    interaction_hi: int,
    vis_threshold: float = 0.5,
    min_points: int = 8,
) -> tuple[list[float], int]:
    """Per-step |Kabsch angle| for t -> t+1 inside [interaction_lo, interaction_hi)."""
    t_lo = max(0, int(interaction_lo))
    t_hi = min(int(tracks_xy.shape[0]) - 1, int(interaction_hi) - 1)
    angles: list[float] = []
    n_skipped = 0
    vis = tracks_vis
    if vis is not None:
        vis = np.asarray(vis)
        if vis.ndim == 3 and vis.shape[-1] == 1:
            vis = vis[..., 0]
    for t in range(t_lo, t_hi):
        p0 = tracks_xy[t]
        p1 = tracks_xy[t + 1]
        if vis is not None:
            m = (vis[t] >= vis_threshold) & (vis[t + 1] >= vis_threshold)
            p0 = p0[m]
            p1 = p1[m]
        if p0.shape[0] < min_points:
            n_skipped += 1
            continue
        ang = _kabsch_2d_abs_angle_deg(p0, p1)
        if math.isfinite(ang):
            angles.append(ang)
        else:
            n_skipped += 1
    return angles, n_skipped


def _cotracker_endpoint_kabsch_deg(
    tracks_xy: np.ndarray,
    tracks_vis: np.ndarray | None,
    interaction_lo: int,
    interaction_hi: int,
    vis_threshold: float = 0.5,
    min_points: int = 8,
) -> float:
    """|Kabsch| between first and last interaction frame (same point indices, both visible)."""
    t0 = max(0, int(interaction_lo))
    t1 = min(int(tracks_xy.shape[0]) - 1, int(interaction_hi) - 1)
    if t1 <= t0:
        return float("nan")
    p0 = tracks_xy[t0]
    p1 = tracks_xy[t1]
    if tracks_vis is not None:
        vis = np.asarray(tracks_vis)
        if vis.ndim == 3 and vis.shape[-1] == 1:
            vis = vis[..., 0]
        m = (vis[t0] >= vis_threshold) & (vis[t1] >= vis_threshold)
        p0 = p0[m]
        p1 = p1[m]
    if p0.shape[0] < min_points:
        return float("nan")
    return _kabsch_2d_abs_angle_deg(p0, p1)


def _rigid_transform_quality_2d(p0: np.ndarray, p1: np.ndarray) -> dict[str, float | int | None]:
    """How much better a 2D rigid transform explains motion than translation.

    Translation-only residual is a good proxy for "the points mostly moved
    together".  A true in-plane rotation should reduce that residual
    substantially when a rigid rotation is also fitted.
    """
    if p0.shape[0] < 2 or p0.shape != p1.shape:
        return {
            "n": int(p0.shape[0]),
            "rotation_gain": None,
            "translation_rms_px": None,
            "rigid_rms_px": None,
            "angle_abs_deg": None,
        }
    c0 = p0.mean(axis=0)
    c1 = p1.mean(axis=0)
    x = p0 - c0
    y = p1 - c1
    delta = (p1 - p0).mean(axis=0)
    trans_res = p1 - (p0 + delta)
    trans_rms = float(np.sqrt(np.mean(np.sum(trans_res * trans_res, axis=1))))

    s = x.T @ y
    u, _, vt = np.linalg.svd(s)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u_adj = u.copy()
        u_adj[:, -1] *= -1.0
        r = u_adj @ vt
    pred_a = x @ r + c1
    pred_b = x @ r.T + c1
    rms_a = float(np.sqrt(np.mean(np.sum((p1 - pred_a) ** 2, axis=1))))
    rms_b = float(np.sqrt(np.mean(np.sum((p1 - pred_b) ** 2, axis=1))))
    rigid_rms = min(rms_a, rms_b)
    gain = 0.0 if trans_rms <= 1e-6 else (trans_rms - rigid_rms) / trans_rms
    return {
        "n": int(p0.shape[0]),
        "rotation_gain": float(np.clip(gain, 0.0, 1.0)),
        "translation_rms_px": trans_rms,
        "rigid_rms_px": rigid_rms,
        "angle_abs_deg": _kabsch_2d_abs_angle_deg(p0, p1),
    }


def _cotracker_transform_quality(
    tracks_xy: np.ndarray,
    tracks_vis: np.ndarray | None,
    t0: int,
    t1: int,
    vis_threshold: float = 0.5,
) -> dict[str, float | int | None]:
    p0 = tracks_xy[t0]
    p1 = tracks_xy[t1]
    if tracks_vis is not None:
        vis = np.asarray(tracks_vis)
        if vis.ndim == 3 and vis.shape[-1] == 1:
            vis = vis[..., 0]
        m = (vis[t0] >= vis_threshold) & (vis[t1] >= vis_threshold)
        p0 = p0[m]
        p1 = p1[m]
    return _rigid_transform_quality_2d(p0, p1)


def _summarize_quality(values: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    gains = [float(v["rotation_gain"]) for v in values if v.get("rotation_gain") is not None]
    ns = [int(v["n"]) for v in values if v.get("n") is not None]
    if not gains:
        return {"n": 0, "rotation_gain_mean": None, "rotation_gain_p50": None, "rotation_gain_p90": None}
    arr = np.asarray(gains, dtype=np.float64)
    return {
        "n": int(len(gains)),
        "support_min": int(min(ns)) if ns else None,
        "rotation_gain_mean": float(np.mean(arr)),
        "rotation_gain_p50": float(np.percentile(arr, 50)),
        "rotation_gain_p90": float(np.percentile(arr, 90)),
    }


def _mask_axis_step_series(
    amodal_masks: np.ndarray,
    interaction_lo: int,
    interaction_hi: int,
) -> tuple[list[float], int]:
    angles: list[float] = []
    n_bad = 0
    t_lo = max(0, int(interaction_lo))
    t_hi = min(int(amodal_masks.shape[0]) - 1, int(interaction_hi) - 1)
    prev = _principal_axis_deg_from_mask(amodal_masks[t_lo])
    if not math.isfinite(prev):
        n_bad += 1
    for t in range(t_lo + 1, t_hi + 1):
        cur = _principal_axis_deg_from_mask(amodal_masks[t])
        if not math.isfinite(cur) or not math.isfinite(prev):
            n_bad += 1
            prev = cur if math.isfinite(cur) else prev
            continue
        step = _wrap_angle_deg(cur - prev)
        angles.append(step)
        prev = cur
    return angles, n_bad


def _mano_param_to_dict(p: Any) -> dict | None:
    if p is None:
        return None
    if isinstance(p, dict):
        return p
    if isinstance(p, np.ndarray):
        if p.shape == () and p.dtype == object:
            x = p.item()
            return x if isinstance(x, dict) else None
    return None


def mano_root_R_camera_series(
    sampled_mano_params: Any,
    interaction_lo: int,
    interaction_hi: int,
) -> np.ndarray | None:
    """Camera-frame MANO global orientation ``R_cam = T_w2c[:3,:3] @ R(root_orient)``.

    Matches the stacking / world-to-camera convention in ``TemporalHandObjectPose``.
    Returns shape ``(hi - lo, 3, 3)`` rotation matrices, or ``None`` if unavailable.
    """
    if _SciRotation is None or sampled_mano_params is None:
        return None
    try:
        n_total = len(sampled_mano_params)
    except TypeError:
        return None
    hi = min(int(interaction_hi), n_total)
    lo = max(0, int(interaction_lo))
    if hi <= lo:
        return None
    mats: list[np.ndarray] = []
    for i in range(lo, hi):
        p = _mano_param_to_dict(sampled_mano_params[i])
        if p is None or "T_w2c" not in p or "root_orient" not in p:
            return None
        twc = np.asarray(p["T_w2c"], dtype=np.float64).reshape(4, 4)
        root_aa = np.asarray(p["root_orient"], dtype=np.float64).reshape(3)
        r_root = _SciRotation.from_rotvec(root_aa).as_matrix()
        r_cam = twc[:3, :3] @ r_root
        mats.append(r_cam.astype(np.float64))
    if not mats:
        return None
    return np.stack(mats, axis=0)


def _pnp_rotation_step_series(
    R_col_major: np.ndarray,
    interaction_lo: int,
    interaction_hi: int,
) -> list[float]:
    t_lo = max(0, int(interaction_lo))
    t_hi = min(int(R_col_major.shape[0]) - 1, int(interaction_hi) - 1)
    out: list[float] = []
    for t in range(t_lo, t_hi):
        out.append(_angle_between_rotations_deg(R_col_major[t], R_col_major[t + 1]))
    return out


def _rotation_geodesic_steps_local(R: np.ndarray) -> list[float]:
    """Consecutive geodesic angles for a contiguous ``(T, 3, 3)`` rotation chunk."""
    if R.ndim != 3 or R.shape[0] < 2:
        return []
    out: list[float] = []
    for t in range(R.shape[0] - 1):
        out.append(_angle_between_rotations_deg(R[t], R[t + 1]))
    return out


def _classify_from_scores(
    cot_mean: float | None,
    cot_p90: float | None,
    mask_mean: float | None,
    mask_max: float | None,
    pnp_mean: float | None,
    cot_cumulative_abs: float | None,
    cot_endpoint_abs: float | None,
    mano_mean: float | None,
    mano_p90: float | None,
    mano_cumulative_steps: float | None,
    mano_endpoint: float | None,
    thr: dict[str, float],
    *,
    cot_endpoint_quality: dict[str, Any] | None = None,
    cot_step_quality: dict[str, Any] | None = None,
) -> str:
    """Return ``rotation_likely`` or ``translation_likely``.

    Downstream policy must choose one branch, so translation is the conservative
    default unless there is a strong rotation cue.  Mask-axis noise and long
    windows of small MANO root jitter should not single-handedly flip the mode.
    """
    def _fin(x: float | None) -> bool:
        return x is not None and math.isfinite(float(x))

    def _quality_gain(q: dict[str, Any] | None) -> float | None:
        if not q:
            return None
        for key in ("rotation_gain", "rotation_gain_p50", "rotation_gain_mean"):
            if key in q and q[key] is not None and math.isfinite(float(q[key])):
                return float(q[key])
        return None

    def _quality_support(q: dict[str, Any] | None) -> int:
        if not q:
            return 0
        for key in ("support_min", "n"):
            if key in q and q[key] is not None:
                return int(q[key])
        return 0

    def _quality_ok(q: dict[str, Any] | None) -> bool:
        gain = _quality_gain(q)
        return (
            gain is not None
            and gain >= thr["cot_rotation_quality_gain_likely"]
            and _quality_support(q) >= int(thr["cot_rotation_quality_min_points"])
        )

    cot_rot_step = False
    if _fin(cot_mean) and _fin(cot_p90):
        cot_rot_step = (
            (cot_mean >= thr["cot_mean_deg_rotation_likely"] or cot_p90 >= thr["cot_p90_deg_rotation_likely"])
            and (_quality_ok(cot_step_quality) or _quality_ok(cot_endpoint_quality))
        )

    cot_quality_ok = _quality_ok(cot_endpoint_quality) or _quality_ok(cot_step_quality)
    rot_integration = False
    if (_fin(cot_cumulative_abs)
            and cot_cumulative_abs >= thr["cot_cumulative_abs_deg_rotation_likely"]
            and cot_quality_ok):
        rot_integration = True
    if (_fin(cot_endpoint_abs)
            and cot_endpoint_abs >= thr["cot_endpoint_abs_deg_rotation_likely"]
            and cot_quality_ok):
        rot_integration = True

    mask_rot = bool(_fin(mask_mean) and mask_mean >= thr["mask_mean_deg_rotation_hint"])
    mask_rot = mask_rot or bool(_fin(mask_max) and mask_max >= thr["mask_max_deg_rotation_hint"])
    pnp_rot = bool(_fin(pnp_mean) and pnp_mean >= thr["pnp_mean_deg_rotation_likely"])

    mano_rot_step = False
    if _fin(mano_mean) and _fin(mano_p90):
        mano_rot_step = (
            mano_mean >= thr["mano_mean_step_deg_rotation_likely"]
            and mano_p90 >= thr["mano_p90_step_deg_rotation_likely"]
        )
    mano_rot_endpoint = bool(_fin(mano_endpoint) and mano_endpoint >= thr["mano_endpoint_geodesic_deg_rotation_likely"])
    mano_rot = mano_rot_step

    strong_rot = rot_integration or cot_rot_step or pnp_rot or mano_rot
    weak_rot_votes = int(mask_rot)
    weak_rot_votes += int(mano_rot_endpoint and (cot_quality_ok or pnp_rot))
    if _fin(mano_cumulative_steps) and _fin(mano_endpoint):
        weak_rot_votes += int(
            mano_cumulative_steps >= thr["mano_cumulative_step_deg_rotation_hint"]
            and mano_endpoint >= 0.75 * thr["mano_endpoint_geodesic_deg_rotation_likely"]
            and (cot_quality_ok or pnp_rot)
        )

    if strong_rot or weak_rot_votes >= 2:
        return "rotation_likely"
    return "translation_likely"


def _summarize_deg(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def build_interaction_motion_profile(
    interaction_lo: int,
    interaction_hi: int,
    tracks_xy: np.ndarray | None,
    tracks_vis: np.ndarray | None,
    amodal_masks: np.ndarray | None,
    pnp_R_col_major: np.ndarray | None = None,
    sampled_mano_params: Any = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable diagnostics dict for the interaction segment.

    Parameters
    ----------
    interaction_lo, interaction_hi
        Sampled-frame indices, half-open ``[lo, hi)`` — **only** the
        ``2-3_interaction`` segment (``approaching_end_idx`` … ``interaction_end_idx``),
        not the ``1-2_approaching`` segment.
    tracks_xy
        CoTracker forward tracks ``(T, N, 2)`` in pixel space.
    tracks_vis
        Optional visibility ``(T, N)`` or ``(T, N, 1)`` in [0, 1].
    amodal_masks
        Optional ``(T, H, W)`` binary or float masks for principal-axis steps.
    pnp_R_col_major
        Optional per-frame rotation from PnP ``(T, 3, 3)`` (column-major / OpenCV).
    sampled_mano_params
        Optional length-``num_sampled_frames`` sequence of per-frame dicts with
        ``root_orient`` and ``T_w2c`` (HaMeR / pipeline JSON), indexed by sampled
        frame index. Used for MANO global root orientation in camera space.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    lo, hi = int(interaction_lo), int(interaction_hi)
    out: dict[str, Any] = {
        "interaction_sampled_half_open": [lo, hi],
        "interaction_range_convention": "[lo, hi) hi exclusive",
        "n_interaction_frames": max(0, hi - lo),
        "thresholds_used": thr,
        "cotracker": None,
        "mask_axis": None,
        "pnp_rotation_steps": None,
        "mano_global_root": None,
        "suggested_mode": "translation_likely",
        "notes": (
            "Heuristic only: per-frame 2D Kabsch misses out-of-plane 3D rotation; "
            "sum(per_step) and first-vs-last Kabsch help. Mask axis is coarse. "
            "MANO root_orient in camera (T_w2c @ R(root_aa)) adds 3D wrist/global cue. "
            "The policy is binary: translation_likely is the conservative default."
        ),
    }

    cot_summary: dict[str, Any] | None = None
    if tracks_xy is not None and tracks_xy.ndim == 3 and hi > lo + 1:
        ang, n_skip = _cotracker_kabsch_series(
            np.asarray(tracks_xy, dtype=np.float64),
            np.asarray(tracks_vis, dtype=np.float64) if tracks_vis is not None else None,
            lo,
            hi,
        )
        _tr = np.asarray(tracks_xy, dtype=np.float64)
        _vis = np.asarray(tracks_vis, dtype=np.float64) if tracks_vis is not None else None
        _end = _cotracker_endpoint_kabsch_deg(_tr, _vis, lo, hi)
        _cum = float(sum(ang)) if ang else float("nan")
        _t0 = max(0, lo)
        _t1 = min(int(_tr.shape[0]) - 1, hi - 1)
        _endpoint_quality = _cotracker_transform_quality(_tr, _vis, _t0, _t1) if _t1 > _t0 else None
        _step_qualities = [
            _cotracker_transform_quality(_tr, _vis, t, t + 1)
            for t in range(max(0, lo), min(int(_tr.shape[0]) - 1, hi - 1))
        ]
        cot_summary = {
            "per_step_abs_deg": _summarize_deg(ang),
            "cumulative_abs_deg_sum": _cum if math.isfinite(_cum) else None,
            "endpoint_first_last_abs_deg": float(_end) if math.isfinite(_end) else None,
            "endpoint_transform_quality": _endpoint_quality,
            "step_transform_quality": _summarize_quality(_step_qualities),
            "n_steps_skipped_low_support": int(n_skip),
        }
        out["cotracker"] = cot_summary

    mask_summary: dict[str, Any] | None = None
    if amodal_masks is not None and hi > lo + 1:
        steps, n_bad = _mask_axis_step_series(np.asarray(amodal_masks), lo, hi)
        mask_summary = {"principal_axis_step_deg": _summarize_deg(steps), "n_axis_eval_failures": int(n_bad)}
        out["mask_axis"] = mask_summary

    pnp_summary: dict[str, Any] | None = None
    if pnp_R_col_major is not None and pnp_R_col_major.ndim == 3 and hi > lo + 1:
        p_steps = _pnp_rotation_step_series(np.asarray(pnp_R_col_major, dtype=np.float64), lo, hi)
        pnp_summary = {"geodesic_step_deg": _summarize_deg(p_steps)}
        out["pnp_rotation_steps"] = pnp_summary

    mano_summary: dict[str, Any] | None = None
    R_mano = mano_root_R_camera_series(sampled_mano_params, lo, hi)
    if R_mano is not None and R_mano.shape[0] >= 2:
        m_steps = _rotation_geodesic_steps_local(R_mano)
        m_cum = float(sum(m_steps)) if m_steps else float("nan")
        m_end = _angle_between_rotations_deg(R_mano[0], R_mano[-1])
        mano_summary = {
            "geodesic_step_deg": _summarize_deg(m_steps),
            "cumulative_step_deg_sum": m_cum if math.isfinite(m_cum) else None,
            "endpoint_first_last_geodesic_deg": float(m_end) if math.isfinite(m_end) else None,
        }
        out["mano_global_root"] = mano_summary

    c_mean = c_p90 = m_mean = m_max = p_mean = c_cum = c_end = None
    h_mean = h_p90 = h_cum = h_end = None
    if cot_summary and cot_summary["per_step_abs_deg"]["mean"] is not None:
        c_mean = float(cot_summary["per_step_abs_deg"]["mean"])
        c_p90 = float(cot_summary["per_step_abs_deg"]["p90"])
    if cot_summary:
        _v = cot_summary.get("cumulative_abs_deg_sum")
        c_cum = float(_v) if _v is not None and math.isfinite(float(_v)) else None
        _e = cot_summary.get("endpoint_first_last_abs_deg")
        c_end = float(_e) if _e is not None and math.isfinite(float(_e)) else None
    if mask_summary and mask_summary["principal_axis_step_deg"]["mean"] is not None:
        m_mean = float(mask_summary["principal_axis_step_deg"]["mean"])
    if mask_summary and mask_summary["principal_axis_step_deg"]["max"] is not None:
        m_max = float(mask_summary["principal_axis_step_deg"]["max"])
    if pnp_summary and pnp_summary["geodesic_step_deg"]["mean"] is not None:
        p_mean = float(pnp_summary["geodesic_step_deg"]["mean"])
    if mano_summary and mano_summary["geodesic_step_deg"]["mean"] is not None:
        h_mean = float(mano_summary["geodesic_step_deg"]["mean"])
        h_p90 = float(mano_summary["geodesic_step_deg"]["p90"])
    if mano_summary:
        _hc = mano_summary.get("cumulative_step_deg_sum")
        h_cum = float(_hc) if _hc is not None and math.isfinite(float(_hc)) else None
        _he = mano_summary.get("endpoint_first_last_geodesic_deg")
        h_end = float(_he) if _he is not None and math.isfinite(float(_he)) else None

    out["suggested_mode"] = _classify_from_scores(
        c_mean,
        c_p90,
        m_mean,
        m_max,
        p_mean,
        c_cum,
        c_end,
        h_mean,
        h_p90,
        h_cum,
        h_end,
        thr,
        cot_endpoint_quality=(cot_summary or {}).get("endpoint_transform_quality") if cot_summary else None,
        cot_step_quality=(cot_summary or {}).get("step_transform_quality") if cot_summary else None,
    )
    return out
